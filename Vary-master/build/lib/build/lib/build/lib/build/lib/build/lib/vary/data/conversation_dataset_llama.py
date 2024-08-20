
import io
import os
import copy
import json
import logging
import torch
import random

from typing import List, Optional, Tuple, Union, Dict, Sequence
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from vary.data.base_dataset import BaseDataset
from vary.utils.constants import *
from vary.utils import conversation as conversation_lib
from vary.utils.box_ops import norm_box_xyxy


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
assistant_format='{content}<|eot_id|>'


class ConversationDataset(BaseDataset):
    """Conversation format dataset stage2 fine-tuning."""

    def __init__(self, datasets, tokenizer, multimodal_cfg):
        super(ConversationDataset, self).__init__(datasets, tokenizer, multimodal_cfg)
        logging.warning("Formatting inputs into conversation type: llama")
        logging.warning("Loading data...")

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

            logging.warning(f"Data from {data_path} provide {len(data)} conversations.")

        assert len(list_data_dict) == len(list_image_path)
        logging.warning(f"{len(list_data_dict)} conversations in total.")
        a_new_list = list(zip(list_data_dict, list_image_path))
        random.shuffle(a_new_list)
        list_data_dict_new, list_image_path_new = zip(*a_new_list)
        self.list_data_dict = list_data_dict_new
        self.list_image_path = list_image_path_new
    
    def multimodal_processor(self, sources):
        for source in sources:
            for sentence in source:
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg['image_token_len']
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = str(sentence["value"]).replace(DEFAULT_IMAGE_TOKEN, replace_token)
        return sources

    def token_processor(self, examples):
        sources = []
        targets = []
        for example in examples:
            assert example[0]["from"] == "human" and example[1]["from"] == "gpt"
            source = system_format.format(content=DEFAULT_SYSTEM_PROMPT) + user_format.format(
                content=example[0]["value"])
            target = assistant_format.format(content=example[1]["value"])

            sources.append(source)
            targets.append(target)

        max_seq_length = self.tokenizer.model_max_length
        tokenized_sources = self.tokenizer(sources, padding="longest", max_length=max_seq_length, truncation=True,
                                           return_attention_mask=False, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, padding="longest", max_length=max_seq_length, truncation=True,
                                           return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return dict(
            input_ids=all_input_ids,
            labels=all_labels,
        )

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # data = self.list_data_dict[i]
        data = copy.deepcopy(self.list_data_dict[i])

        if isinstance(data, dict):
            if 'image' in data:
                image_path = self.list_image_path[i]
                image_file = data['image']

                try:
                    image = Image.open(image_path + image_file).convert('RGB')
                    w, h = image.size
                except:
                    print(f'cannot identify image file {image_path + image_file}.')
                    return self.__getitem__(0)

                if '<box>' in data['conversations'][0]['value'] or '<box>' in data['conversations'][1]['value'] or \
                    '<point>' in data['conversations'][0]['value'] or '<point>' in data['conversations'][1]['value']:
                    mode = 0
                else:
                    mode = 1

                try:
                    image, image_1 = self.image_processor(image, mode)
                except:
                    print(f'image {image_file} are broken or grayscale! we thus select 0-th sample instead!')
                    return self.__getitem__(0)

            conversations = self.multimodal_processor([data["conversations"]])

        else:
            conversations = [data]

        # align with fastchat & llava here, put the conversation into a list for tokenization
        data_dict = self.token_processor(conversations)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        
        if isinstance(data, dict) and 'image' in data:
            data_dict['image'] = [image]
            data_dict['image_high'] = [image_1]
        else:
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            data_dict['image_high'] = [torch.zeros(3, 1024, 1024)]

        if 'boxes' in data:
            norm = data.get('norm', False)
            if norm:
                normalized_boxes = data['boxes']
            else:
                normalized_boxes = []
                for box in data['boxes']:
                    normalized_boxes.append(norm_box_xyxy(box, w=w, h=h))
            data_dict['loc_inputs'] = normalized_boxes
            data_dict['loc_targets'] = normalized_boxes
        return data_dict

