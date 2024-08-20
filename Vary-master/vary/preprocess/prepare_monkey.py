import os
import json
import re
import random
import shutil
import sys
import cv2
import transformers
import multiprocessing
from glob import glob
from tqdm import tqdm
from vary.utils import conversation as conversation_lib


def preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
):
    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    if roles[source[0]["from"]] != conv.roles[0]:
        source = source[1:]
    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2]
        value = str(sentence["value"]).replace('<image>', replace_token)
        conv.append_message(role, value)
    conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=4096,
        truncation=True,
        add_special_tokens=False
    ).input_ids

    return input_ids[0]


def process(datas, tokenizer, progress_queue):
    for data in datas:
        path = os.path.join('/mnt/ceph2/datasets/monkey/TextMonkey_data/train_images', data['id'][5:])
        if not os.path.exists(path):
            # print(data['id'])
            progress_queue.put(('miss', data['id']))
            continue
        try:
            roles = {"user": "human", "assistant": "gpt"}
            source = []
            for sentence in data["conversations"]:
                if '<img>' in sentence["value"]:
                    sentence["value"] = re.sub(r'<img>.*?</img>', '<image>', sentence["value"])
                source.append({"from": roles[sentence["from"]], "value": sentence["value"]})
            input_id = preprocess(source, tokenizer)
            if len(input_id) > 3600:
                progress_queue.put(('toolong', path))
            else:
                progress_queue.put(('success', {"image": path.replace('/mnt/ceph2/datasets/monkey/TextMonkey_data/', ''),
                                                "conversations": source}))
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue


def process_parallel(target, datas, tokenizer, progress_queue, output_path):
    chunk_size = len(datas) // multiprocessing.cpu_count()
    if chunk_size == 0:
        chunk_size = len(datas)
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [datas[i:i + chunk_size] for i in range(0, len(datas), chunk_size)]
    runners = [multiprocessing.Process(target=target, args=(chunk, tokenizer, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(datas))
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'miss': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'],
                                     miss=counts['miss'])
        except Exception as e:
            if all(not runner.is_alive() for runner in runners):
                break
            continue
        progress_bar.update(1)
    progress_bar.close()
    print(len(conversations), counts)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    print('done')


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True

    datas = json.load(open('/mnt/ceph2/datasets/monkey/TextMonkey_data/stage1_qa.json', encoding='utf-8'))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, datas, tokenizer, progress_queue, '/mnt/ceph2/datasets/monkey/conversations.json')
    else:
        process(datas, tokenizer, progress_queue)


def check():
    datas = json.load(open('/mnt/ceph2/datasets/monkey/TextMonkey_data/stage1_qa.json', encoding='utf-8'))
    for data in datas:
        if 'cdip' in data['id']:
            continue
        if 'cdip' in data['id']:
            path = os.path.join('/mnt/ceph2/datasets/monkey/TextMonkey_data/more_images', data['id'][5:])
        else:
            path = os.path.join('/mnt/ceph2/datasets/monkey/TextMonkey_data/train_images', data['id'][5:])
        if not os.path.exists(path):
            print(data['id'])
            break
        print(data['conversations'])
        img = cv2.imread(path)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()


