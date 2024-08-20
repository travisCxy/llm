from typing import Iterable, List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
from PIL import Image
from functools import lru_cache
from transformers import CLIPVisionConfig, Qwen2Config, AutoConfig
from vary.model.vision_encoder.sam import build_sam_vit_l

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig, LoRAConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.clip import CLIPVisionModel
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, BatchedTensors
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import (cached_get_tokenizer,
                                   repeat_and_pad_image_tokens)
from vllm.sequence import IntermediateTensors, SamplerOutput, SequenceData
from vllm.model_executor.models.interfaces import SupportsVision, SupportsLoRA


_KEYS_TO_MODIFY_MAPPING = {
    "model.lm_head": "lm_head",
    "model.model": "model",
    "model.vision_tower": "vision_tower",
    "model.vision_tower_high": "vision_tower_high",
    "model.mm_projector": "mm_projector",
    "model.mm_projector_vary": "mm_projector_vary",
}


def get_image_processor(path, image_size):
    from transformers import CLIPImageProcessor
    from vary.model.plug.blip_process import BlipImageEvalProcessor
    image_processor = CLIPImageProcessor.from_pretrained(path, torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=image_size)
    return image_processor, image_processor_high


cached_get_image_processor = lru_cache(get_image_processor)


class Qwen2ModelForMultiModal(Qwen2Model):
    def __init__(self, config: Qwen2Config, cache_config: CacheConfig | None = None,
                 quant_config: QuantizationConfig | None = None) -> None:
        super().__init__(config, cache_config, quant_config)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata, inputs_embeds: torch.Tensor) -> torch.Tensor:
        if input_ids is not None and inputs_embeds is None:

            return super().forward(input_ids, positions, kv_caches, attn_metadata)
        elif input_ids is None and inputs_embeds is not None:

            hidden_states = inputs_embeds
            residual = None
            for i in range(len(self.layers)):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    kv_caches[i],
                    attn_metadata,
                    residual,
                )
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        else:
            raise Exception("impossible!")


class VaryImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data1: torch.Tensor
    data2: torch.Tensor
    """Shape: `(batch_size, num_channels, height, width)`"""


VaryImageInputs = VaryImagePixelInputs


class varyConfig(Qwen2Config):
    model_type = "vary"


AutoConfig.register("vary", varyConfig)


def get_max_vary_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(varyConfig)
    return hf_config.image_token_len


def dummy_data_for_vary(ctx: InputContext, seq_len: int):
    hf_config = ctx.get_hf_config(varyConfig)

    token_ids = [hf_config.im_start_token] + [hf_config.im_patch_token] * hf_config.image_token_len + [hf_config.im_end_token]
    token_ids += [0] * (seq_len - len(token_ids))
    seq_data = SequenceData(token_ids)

    image = Image.new("RGB", (1024, 1024), color=0)
    mm_data = {"image": image}
    return seq_data, mm_data


def input_mapper_for_vary(ctx: InputContext, data: object):
    assert isinstance(data, Image.Image)
    if data.mode != "RGB":
        data = data.convert("RGB")
    image_processor, image_processor_high = cached_get_image_processor("/cache/vit-large-patch14/", 1024)
    image_tensor_1 = image_processor.preprocess(data, return_tensors='pt')['pixel_values']
    image_tensor_2 = image_processor_high(data).unsqueeze(0)
    return MultiModalInputs({"image1": image_tensor_1, "image2": image_tensor_2})


def input_processor_for_vary(ctx: InputContext, llm_inputs: LLMInputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(varyConfig)

    tokenizer = cached_get_tokenizer(model_config.tokenizer)

    new_prompt, new_token_ids = repeat_and_pad_image_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        image_token_id=hf_config.im_patch_token,
        repeat_count=hf_config.image_token_len,
        pad_token_left=hf_config.im_start_token,
        pad_token_right=hf_config.im_end_token,
    )

    # NOTE: Create a defensive copy of the original inputs
    return LLMInputs(prompt_token_ids=new_token_ids,
                     prompt=new_prompt,
                     multi_modal_data=multi_modal_data)


@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_vary)
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_vary_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_vary)
@INPUT_REGISTRY.register_input_processor(input_processor_for_vary)
class VaryQwen2ForConditionalGeneration(nn.Module, SupportsVision, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
        # "lm_head",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self,
                 config: varyConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None,) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config
        self.lora_config = lora_config

        # TODO: Optionally initializes this for supporting embeddings.
        vision_config = AutoConfig.from_pretrained(config.vision_tower).vision_config
        vision_feature_layer = config.vision_select_layer
        if vision_feature_layer < 0:
            num_hidden_layers = vision_config.num_hidden_layers + vision_feature_layer + 1
        else:
            num_hidden_layers = vision_feature_layer + 1
        self.vision_tower = CLIPVisionModel(vision_config, num_hidden_layers_override=num_hidden_layers)
        self.vision_tower_high = build_sam_vit_l()
        self.mm_projector = nn.Linear(1024, 1792)
        self.mm_projector_vary = nn.Linear(1024, 1792)

        self.quant_config = quant_config
        self.model = Qwen2ModelForMultiModal(config, cache_config, quant_config)
        self.unpadded_vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=self.model.vocab_size,
                quant_config=quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def _validate_pixel_values(self, data: torch.Tensor, image_size) -> torch.Tensor:
        h = w = image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[VaryImageInputs]:
        image1 = kwargs.pop("image1", None)
        image2 = kwargs.pop("image2", None)

        if image1 is None or image2 is None:
            return None
        # print(image1.shape, image2.shape)

        if not isinstance(image1, torch.Tensor):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(image1)}")
        if not isinstance(image2, torch.Tensor):
            raise ValueError("Incorrect type of pixel values. "
                             f"Got type: {type(image2)}")

        return VaryImagePixelInputs(
            type="pixel_values",
            data1=self._validate_pixel_values(image1, 224),
            data2=self._validate_pixel_values(image2, 1024),
        )

    def _select_image_features(self, image_features: torch.Tensor, *,
                               strategy: str) -> torch.Tensor:
        # Copied from https://github.com/huggingface/transformers/blob/39c3c0a72af6fbda5614dde02ff236069bb79827/src/transformers/models/vary/modeling_vary.py#L421  # noqa
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(self, vision_tower: CLIPVisionModel,
                                  pixel_values: torch.Tensor) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values,
                                      self.config.vision_feature_layer)

        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

    def _process_image_pixels(self,
                              inputs: VaryImagePixelInputs) -> torch.Tensor:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        return self._image_pixels_to_features(self.vision_tower, pixel_values)

    def _process_image_input(self,
                             image_input: VaryImageInputs) -> torch.Tensor:
        image1 = image_input["data1"].to(torch.float16)
        image2 = image_input["data2"].to(torch.float16)

        vision_select_layer = getattr(self.config, "vision_select_layer", -1)
        image_features = self.vision_tower(image1)
        # select_hidden_state = image_forward_out.hidden_states[vision_select_layer]
        image_feature = image_features[:, 1:]

        cnn_feature = self.vision_tower_high(image2)
        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)

        image_features_1 = self.mm_projector(image_feature)
        image_features_2 = self.mm_projector_vary(cnn_feature)
        image_features = torch.cat((image_features_1, image_features_2), dim=-1)

        return image_features

    def merge_vision_embeddings(self, input_ids: torch.Tensor,
                                inputs_embeds: torch.Tensor,
                                vision_embeddings: BatchedTensors) -> torch.Tensor:
        mask = (input_ids == self.config.im_patch_token)
        num_expected_tokens = mask.sum()

        if isinstance(vision_embeddings, torch.Tensor):
            batch_size, batch_tokens, *_, embed_dim = vision_embeddings.shape
            total_tokens = batch_size * batch_tokens
            if num_expected_tokens != total_tokens:
                expr = f"{batch_size} x {batch_tokens}"
                raise ValueError(
                    f"Attempted to assign {expr} = {total_tokens} "
                    f"image tokens to {num_expected_tokens} placeholders")

            inputs_embeds[mask] = vision_embeddings.view(total_tokens, embed_dim)
        else:
            size_per_batch = [t.shape[0] for t in vision_embeddings]
            total_tokens = sum(size_per_batch)
            if num_expected_tokens != total_tokens:
                expr = ' + '.join(map(str, size_per_batch))
                raise ValueError(
                    f"Attempted to assign {expr} = {total_tokens} "
                    f"image tokens to {num_expected_tokens} placeholders")

            inputs_embeds[mask] = torch.cat(vision_embeddings)

        return inputs_embeds


    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            **kwargs: object,
    ) -> SamplerOutput:
        """Run forward pass for LLaVA-1.5.

        One key thing to understand is the `input_ids` already accounts for the
        positions of the to-be-inserted image embeddings.

        Concretely, consider a text prompt:
        `"USER: <image>\\nWhat's the content of the image?\\nASSISTANT:"`.

        Tokenizer outputs:
        `[1, 3148, 1001, 29901, 29871, 32000, 29871, 13, 5618, 29915, 29879,
        278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566, 29901]`.

        To reserve space in KV cache, we have to insert placeholder tokens
        before they are inputted to the model, so the input processor prepends
        additional image tokens (denoted as `32000`), resulting in:
        `[1, 3148, 1001, 29901, 29871, 32000, ..., 32000, 29871, 13, 5618,
        29915, 29879, 278, 2793, 310, 278, 1967, 29973, 13, 22933, 9047, 13566,
        29901]`.

        We insert 575 tokens so that including the original image token in the
        input, there are a total of 576 (24 * 24) image tokens, which
        corresponds to the number of image tokens inputted to the language
        model, i.e. the number of image tokens outputted by the visual encoder.

        This way, the `positions` and `attn_metadata` are consistent
        with the `input_ids`.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values: The pixels in each input image.

        See also:
            :class:`VaryImageInputs`
        """
        # print('input_ids:', input_ids.shape)
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            vision_embeddings = self._process_image_input(image_input)
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = self.merge_vision_embeddings(input_ids, inputs_embeds, vision_embeddings)
            # print('inputs_embeds:', inputs_embeds.shape)
            input_ids = None
        else:
            inputs_embeds = None

        hidden_states = self.model(input_ids,
                                   positions,
                                   kv_caches,
                                   attn_metadata,
                                   inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # only doing this for language model part for now.
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        params_dict_keys = dict(params_dict)
        # print('-------------------------------------------------')
        # for key in params_dict_keys.keys():
        #     print(key)
        # print('-------------------------------------------------')
        # for name, loaded_weight in weights:
        #     print(name)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # post_layernorm is not needed in CLIPVisionModel
            if "vision_model.post_layernorm" in name:
                continue
            for key_to_modify, new_key in _KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in name:
                    name = name.replace(key_to_modify, new_key)
            use_default_weight_loading = True
            if "vision" not in name:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    params_dict_keys.pop(name, None)
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    use_default_weight_loading = False
                    break
                # else:
                #    use_default_weight_loading = True
            if use_default_weight_loading:
                if name not in params_dict:
                    # if name.endswith('lm_head.weight'):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if param.size() != loaded_weight.size():
                    print(name, param.size(), loaded_weight.size())
                weight_loader(param, loaded_weight)
                params_dict_keys.pop(name)
        print(params_dict_keys.keys())
