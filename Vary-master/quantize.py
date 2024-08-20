import json
import random
import copy
import torch
from PIL import Image
from vary.model import *
from vary.data import make_supervised_data_module
from vary.utils.arguments import *
from vary.utils.constants import *
from vary.utils import conversation as conversation_lib
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from torch.utils.data import Dataset, DataLoader
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig, Quantizer
# from optimum.gptq import GPTQQuantizer, load_quantized_model
from auto_gptq.modeling import BaseGPTQForCausalLM


class QwenGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "QWenBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = [
        "transformer.wte",
        "transformer.wpe",
        "transformer.ln_f",
        "transformer.visual",
    ]
    inside_layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.w1", "mlp.w2"],
        ["mlp.c_proj"],
    ]


class VaryDataset(Dataset):
    def __init__(self, datasets, tokenizer):
        conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        print("Formatting inputs into conversation type: mpt-fixed")
        print("Loading data...")

        list_data_dict = []
        list_image_path = []

        for name in datasets.split("+"):
            # for name in vary_data_dict[name_all]:
            dataset = CONVERSATION_DATA[name]

            data_path = dataset['annotations']
            data = json.load(open(data_path, "r"))

            list_data_dict.extend(data)

            image_path = dataset['images']

            list_image_path.extend([image_path] * len(data))

            print(f"Data from {data_path} provide {len(data)} conversations.")

        assert len(list_data_dict) == len(list_image_path)
        a_new_list = list(zip(list_data_dict, list_image_path))
        random.shuffle(a_new_list)
        list_data_dict_new, list_image_path_new = zip(*a_new_list)
        self.list_data_dict = list_data_dict_new
        self.list_image_path = list_image_path_new
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.list_data_dict)

    def multimodal_processor(self, sources):
        for source in sources:
            for sentence in source:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources

    def token_processor(self, sources):
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=False
        ).input_ids

        return input_ids

    def __getitem__(self, i):
        data = copy.deepcopy(self.list_data_dict[i])

        image_path = self.list_image_path[i]
        image_file = data['image']
        image = Image.open(image_path + image_file).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_1 = image_processor_high(image)
        images = list(zip(image_tensor, image_tensor_1))

        conversations = self.multimodal_processor([data["conversations"]])
        input_ids = self.token_processor(conversations)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images
        )


def calibration_data_generator(dataloader):
    for batch in dataloader:
        yield batch


if __name__ == "__main__":
    model_name_or_path = '/mnt/ceph2/Vary/runs/0524/'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, padding_side="right", model_max_length=4096,)
    # model = varyQwenForCausalLM.from_pretrained(model_name_or_path)
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    dataset = VaryDataset(datasets="sjb_det14_train", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    model = QwenGPTQForCausalLM.from_pretrained(model_name_or_path, quantize_config)
    model.quantize(calibration_data_generator(dataloader))
    model.save_quantized('quantized_model', use_safetensors=True)

    # data = []
    # i = 10
    # for batch in dataloader:
    #     data.append(batch)
    #     i -= 1
    #     if i == 0:
    #         break
    # quantizer = GPTQQuantizer(bits=4, dataset=data, block_name_to_quantize="model.decoder.layers", model_seqlen=4096)
    # quantized_model = quantizer.quantize_model(model, tokenizer)
    # quantizer.save(quantized_model, 'quantized_model')
