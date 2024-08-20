import os
import json
import re
import random
import sys
import cv2
import transformers
import multiprocessing
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


def process(chunk, tokenizer, progress_queue):
    for item in chunk:
        path = os.path.join('/mnt/ceph2/datasets/', item['image'])
        assert os.path.exists(path)
        text1 = item['conversations'][0]['value']
        text2 = item['conversations'][1]['value']
        ocr = ''
        lines = text1.split('\n')
        for line in lines:
            if '<ref' in line:
                ocr += line
        text1 = f'<image>\nOCR and {lines[-1]}\n'
        text2 = 'OCR:\n' + ocr + '\n' + lines[-1] + '\n' + text2
        source = {"image": item['image'],
                  "conversations":
                      [{"from": "human", "value": text1},
                       {"from": "gpt", "value": text2}]}
        input_id = preprocess(source, tokenizer)
        if len(input_id) > 3600:
            progress_queue.put(('toolong', path))
            continue
        progress_queue.put(('success', source))


def process_parallel(files, tokenizer, progress_queue, output_path):
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, tokenizer, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
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


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Qwen/Qwen2-7B-Instruct/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    tokenizer.add_tokens("</s>", special_tokens=True)
    tokenizer.add_tokens('<imgpad>', special_tokens=True)
    tokenizer.add_tokens('<img>', special_tokens=True)
    tokenizer.add_tokens('</img>', special_tokens=True)
    tokenizer.add_tokens('<box>', special_tokens=True)
    tokenizer.add_tokens('</box>', special_tokens=True)
    tokenizer.add_tokens('<ref>', special_tokens=True)
    tokenizer.add_tokens('</ref>', special_tokens=True)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True

    progress_queue = multiprocessing.Queue()
    datas = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det14.json', encoding='utf-8'))
    if parallel:
        process_parallel(datas, tokenizer, progress_queue,'/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det_ocr.json')
    else:
        process(datas, tokenizer, progress_queue)


def check():
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det_ocr.json', encoding='utf-8'))
    for c in tqdm(conversations):
        path = os.path.join('/mnt/ceph2/datasets/', c['image'])
        print(path)
        print(c['conversations'][0]['value'])
        print(c['conversations'][1]['value'])
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', c['conversations'][1]['value'])
        for match in matches:
            x0, y0, x1, y1 = map(int, match)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    # prepare()
    check()
