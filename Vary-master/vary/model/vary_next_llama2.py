from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
    CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from vary.utils.constants import *
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from vary.model.vision_encoder.sam import build_sam_vit_b
from vary.utils.box_ops import generalized_box_iou, box_iou
from vary.utils.constants import *
import deepspeed

global_step = 0


class varyConfig(LlamaConfig):
    model_type = "vary"


class varyQwenModel(LlamaModel):
    config_class = varyConfig

    def __init__(self, config: LlamaConfig):
        super(varyQwenModel, self).__init__(config)

        # self.vision_tower_high = build_sam_vit_b()

        self.mm_projector_vary = nn.Linear(1024, 4096)

    def initialize_vision_modules(
            self,
            vision_tower,
            pretrained_stage1_model=None,
            freeze_vision_tower=False,
            use_im_start_end=False,
            vision_select_layer=-1,
            dtype=torch.float16,
            device="cuda"
    ):

        self.vision_tower_high = build_sam_vit_b(checkpoint='/mnt/ceph/pretrained/Ucas/vision_tower_high.pth')

        # 224*224
        image_processor = CLIPImageProcessor.from_pretrained('/mnt/ceph/pretrained/Ucas/vit-large-patch14/')
        # 1024*1024
        image_processor_high = BlipImageEvalProcessor(image_size=1024)

        self.vision_tower_high = self.vision_tower_high.to(dtype=dtype, device=device)
        self.mm_projector_vary = self.mm_projector_vary.to(dtype=dtype, device=device)

        image_token_len = 256

        self.config.image_token_len = image_token_len
        self.config.use_im_start_end = True
        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower

        return dict(
            image_processor=image_processor,
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
            # vision_config=vision_config
        )

    def encode_input_embeds(self, input_ids, images, loc_embeds, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower_high = getattr(self, 'vision_tower_high', None)

        if vision_tower_high is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:

            use_im_start_end = getattr(self.config, "use_im_start_end", -1)

            vision_select_layer = getattr(self.config, "vision_select_layer", -1)
            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)
            freeze_vision_tower = getattr(self.config, "freeze_vision_tower", False)

            # im_patch_token = 151859
            # im_start_token = 151857
            # im_end_token = 151858

            image_features = []
            for image in images:
                with torch.set_grad_enabled(False):
                    cnn_feature = vision_tower_high(image[1])
                    cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)  # 256*1024
                image_features.append(cnn_feature)

            if type(images) is list:
                image_features = [self.mm_projector_vary(image_feature) for image_feature in image_features]
            else:
                raise NotImplementedError

            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector_vary(dummy_image_features)
            use_im_start_end = True
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")

                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_image_features.shape[0]

                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")

                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos + 1],
                                per_cur_image_features,
                                cur_input_embeds[image_start_token_pos + num_patches + 1:]
                            ),
                            dim=0
                        )

                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        # add the loc embeddings into the input_embeds
        if (input_ids == self.config.boxes_token).sum() > 0 and loc_embeds is not None:
            inputs_embeds[input_ids == self.config.boxes_token] = loc_embeds.type(inputs_embeds.dtype)

        return inputs_embeds

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            loc_embeds: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if inputs_embeds is None:
            inputs_embeds = self.encode_input_embeds(input_ids, images, loc_embeds, inputs_embeds)

        return super(varyQwenModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class varyQwenForCausalLM(LlamaForCausalLM):
    config_class = varyConfig

    # supports_gradient_checkpointing = True

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = varyQwenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.loc_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

        self.loc_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 4)
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            loc_inputs=None,
            loc_targets=None,

    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if labels is not None:
            labels[labels == self.config.boxes_token] = -100

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO change to loc_inputs
        loc_embeds = None
        if loc_inputs is not None and len(loc_inputs) > 0:
            loc_embeds = self.loc_encoder(loc_inputs)

        transformer_outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            images=images,
            return_dict=return_dict

        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        # logits

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        box_loss = None
        cycle_loss1 = None
        if loc_targets is not None and len(loc_targets) > 0:
            last_hidden_states = transformer_outputs.hidden_states[-1]
            last_hidden_states = last_hidden_states.view(-1, last_hidden_states.size(-1))
            loc_positions = ((input_ids.flatten() == self.config.at_token)
                             & (labels.flatten() > 0)).nonzero().flatten()
            selected_hidden_states = last_hidden_states[loc_positions]
            pred_locs = self.loc_decoder(selected_hidden_states)
            box_loss = self.box_loss(pred_locs, loc_targets)
            loss += box_loss

            # cycle loss
            pred_output_embeds = self.loc_encoder(pred_locs)
            cycle_loss1 = F.mse_loss(pred_output_embeds, selected_hidden_states, reduction="none")
            cycle_loss1 = self.masked_loss(cycle_loss1, 1)
            loss += cycle_loss1
            # print()

        # cycle loss
        if loc_embeds is not None:
            pred_input_locs = self.loc_decoder(loc_embeds)
            cycle_loss2 = F.l1_loss(pred_input_locs, loc_inputs, reduction="none")
            cycle_loss2 = self.masked_loss(cycle_loss2, 1)
            loss += cycle_loss2

            global global_step
            global_step += 1
            if deepspeed.comm.get_rank() == 0 and global_step % 10 == 0:
                print('loss:', loss.item(),
                      'box_loss:', box_loss.item(),
                      'box_iou:', torch.diag(box_iou(pred_locs, loc_targets)[0]).mean().item(),
                      'cycle_loss1:', cycle_loss1.item(),
                      'cycle_loss2:', cycle_loss2.item())

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def box_loss(self, src_boxes, target_boxes):
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = self.masked_loss(loss_bbox, 1)

        mask = (src_boxes[:, 2:] >= src_boxes[:, :2]).all(-1)
        src_boxes = src_boxes[mask]
        target_boxes = target_boxes[mask]
        # if not mask.all():
        #     print(len(mask)-mask.sum())

        loss_giou = 1 - torch.diag(generalized_box_iou(
            src_boxes,
            target_boxes))
        loss_giou = self.masked_loss(loss_giou, 1)
        return loss_bbox*2 + loss_giou/5

    def masked_loss(self, loss, n):
        mask = torch.ones_like(loss)
        mask[-n:] = 1e-10
        loss = (loss*mask).sum()/(mask.sum())
        return loss

    def initialize_vision_tokenizer(
            self,
            tokenizer,
            freeze_lm_model=False,
            pretrained_stage1_model=None,
            device="cuda"
    ):
        config = self.get_model().config

        tokenizer.add_tokens("</s>", special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        tokenizer.add_tokens(DEFAULT_IMAGE_PATCH_TOKEN, special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
        config.im_patch_token = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        config.use_im_start_end = True

        if config.use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_BOXES_TOKEN, DEFAULT_AT_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            config.im_start_token, config.im_end_token, config.boxes_token, config.at_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_BOXES_TOKEN, DEFAULT_AT_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

    def generate(
        self,
        input_ids=None,
        images=None,
        do_sample=False,
        num_beams=1,
        # temperature=0.2,
        streamer=None,
        max_new_tokens=2048,
        stopping_criteria=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        loc_inputs = kwargs.pop("loc_inputs", None)
        loc_embeds = None
        if loc_inputs is not None and len(loc_inputs) > 0:
            loc_embeds = self.loc_encoder(loc_inputs.type(dtype))
            vision_tower = self.model.get_vision_tower()
            num = (input_ids == self.config.boxes_token).sum()
            loc_embeds = loc_embeds[:num]
            if num == 0:
                loc_embeds = None

        input_embeds = self.model.encode_input_embeds(input_ids, images, loc_embeds, inputs_embeds=None)

        outputs = super(varyQwenForCausalLM, self).generate(
            # input_ids=input_ids,
            # images=images,
            inputs_embeds=input_embeds,
            # attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            min_length=1,
            top_p=kwargs.get("top_p", 0.8),
            repetition_penalty=1.0,
            length_penalty=1,
            temperature=kwargs.get("temperature", 0.75),
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
            top_k=kwargs.get("top_k", 5),
        )

        loc_hidden_states = []
        if hasattr(outputs, "beam_indices"):  # beam size > 1
            loc_ids = (outputs.sequences == self.config.at_token).nonzero()
            hidden_states = outputs.hidden_states
            beam_indices = outputs.beam_indices

            for lid in loc_ids:
                # assign to box
                outputs.sequences[lid[0], lid[1] + 1] = self.config.boxes_token
                beam_idx = beam_indices[lid[0], lid[1]]
                loc_h = hidden_states[lid[1]][-1][beam_idx]
                loc_hidden_states.append(loc_h.squeeze())
            if len(loc_hidden_states) > 0:
                loc_hidden_states = torch.stack(loc_hidden_states)
        else:  # beam_size == 1
            loc_ids = (outputs.sequences == self.config.at_token).nonzero()
            hidden_states = outputs.hidden_states
            for lid in loc_ids:
                outputs.sequences[lid[0], lid[1] + 1] = self.config.boxes_token
                loc_h = hidden_states[lid[1]][-1]
                loc_hidden_states.append(loc_h.squeeze())
            if len(loc_hidden_states) > 0:
                loc_hidden_states = torch.stack(loc_hidden_states)

        pred_locs = None
        if len(loc_hidden_states) > 0:
            # loc_hidden_states = loc_hidden_states.type(dtype)
            pred_locs = self.loc_decoder(loc_hidden_states)
        return outputs.sequences, pred_locs


    def prepare_inputs_for_generation(
            self, input_ids, loc_inputs=None,
            past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            loc_ids = None
            if input_ids.size(-1)>=2:
                loc_ids = input_ids[:, -2]
            input_ids = input_ids[:, -1:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            inputs_embeds = self.model.embed_tokens(input_ids)
            hidden_states = kwargs.pop("hidden_states", None)
            # need to incorporate location information
            if loc_ids is not None and (loc_ids==self.config.at_token).any():
                mask = loc_ids==self.config.at_token
                loc_embeds = hidden_states[-1][mask][:, -1:, :]
                loc_embeds = loc_embeds.type(inputs_embeds.dtype)

                pred_locs = self.loc_decoder(loc_embeds)
                loc_embeds = self.loc_encoder(pred_locs)
                inputs_embeds[mask] = loc_embeds
            model_inputs = {"inputs_embeds": inputs_embeds}
            # model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder=False,
        standardize_cache_format=False,
    ):
        model_kwargs = super(varyQwenForCausalLM, self)._update_model_kwargs_for_generation(outputs,
                                                                                model_kwargs,
                                                                                is_encoder_decoder,
                                                                                standardize_cache_format)
        model_kwargs.update({"hidden_states": outputs.hidden_states})
        return model_kwargs


AutoConfig.register("vary", varyConfig)
AutoModelForCausalLM.register(varyConfig, varyQwenForCausalLM)
