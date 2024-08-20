import os
import json
from tqdm import tqdm
from vary.utils.constants import CONVERSATION_DATA
import transformers


def preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
):
    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'
    text1 = source["conversations"][0]["value"]
    text2 = source["conversations"][1]["value"]
    text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
    tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)

    return tokenized.input_ids[0]


if __name__ == "__main__":
    # tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Qwen/Qwen2-7B-Instruct/',
    #                                                        trust_remote_code=True,
    #                                                        padding_side="right",
    #                                                        model_max_length=4096)
    # tokenizer.add_tokens("</s>", special_tokens=True)
    # tokenizer.add_tokens('<imgpad>', special_tokens=True)
    # tokenizer.add_tokens('<img>', special_tokens=True)
    # tokenizer.add_tokens('</img>', special_tokens=True)
    # tokenizer.add_tokens('<box>', special_tokens=True)
    # tokenizer.add_tokens('</box>', special_tokens=True)
    # tokenizer.add_tokens('<ref>', special_tokens=True)
    # tokenizer.add_tokens('</ref>', special_tokens=True)
    # print('tokenizer:', tokenizer.__class__.__name__)

    for key in 'box_det_train+box_qms_train'.split('+'):
        print(key)
        annotations = CONVERSATION_DATA[key]['annotations']
        images = CONVERSATION_DATA[key]['images']
        conversations = json.load(open(annotations, encoding='utf-8'))
        for c in tqdm(conversations):
            path = os.path.join(images, c['image'])
            if not os.path.exists(path):
                print(path)
                break
            # input_id = preprocess(c, tokenizer)
            # if len(input_id) > 3600:
            #     print(len(input_id), path)
