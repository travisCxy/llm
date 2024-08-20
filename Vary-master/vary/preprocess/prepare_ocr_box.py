import os
import json
import re
import random
import sys
import shutil
import transformers
import multiprocessing
from glob import glob
from tqdm import tqdm


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


def process(data, tokenizer, progress_queue):
    for d in data:
        path = d['image']
        try:
            text1 = d['conversations'][0]['value']
            text2 = d['conversations'][1]['value']
            text2 = re.sub(r'\n+', '\n', text2)
            matches = re.findall(r'<box>\(\d+,\d+\),\(\d+,\d+\)</box>', text2)
            for match in matches:
                text2 = text2.replace(match, '<ref><pic></ref>' + match)
            source = {"image": path,
                      "conversations":
                          [{"from": "human", "value": text1},
                           {"from": "gpt", "value": text2}]}
            input_id = preprocess(source, tokenizer)
            if len(input_id) > 3600:
                progress_queue.put(('toolong', path))
                continue
            progress_queue.put(('success', source))
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue


def process_parallel(data, tokenizer, progress_queue, output_path):
    chunk_size = len(data) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, tokenizer, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(data))
    counts = {'success': 0, 'error': 0, 'toolong': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'])
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


def prepare_tiku(tokenizer, parallel):
    data = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_ocr_box_1M.json', encoding='utf-8'))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(data, tokenizer, progress_queue, '/mnt/ceph2/datasets/tiku/ocr/conversations_tiku_ocr_box_1M.json')
    else:
        process(data, tokenizer, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-llava80k/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    prepare_tiku(tokenizer, parallel)


def check():
    import cv2
    # conversations = json.load(open('/mnt/ceph2/datasets/tiku/ocr/conversations_tiku_ocr_box_1M.json', encoding='utf-8'))
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_ocr_box_1M.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        text = c['conversations'][1]['value']
        if '<img' not in text:
            continue
        path = '/mnt/ceph2/datasets/tiku/' + c['image']
        print(path)
        print(c['conversations'][0]['value'])
        print(text)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        matches = re.findall(r'(<img x=(-?\d\.\d+) y=(-?\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text)
        print(matches)
        for match in matches:
            x, y, w, h = map(float, match[1:])
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            x *= 1000
            y *= 1000
            w *= 1000
            h *= 1000
            x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', c['conversations'][1]['value'])
        for match in matches:
            x0, y0, x1, y1 = map(int, match)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()
