import os
import json
import cv2
import random
import sys
import shutil
import transformers
import multiprocessing
import rotated_rect_utils
from glob import glob
from tqdm import tqdm


def find_topic_item(topic_items, sub_item, thresh=0.7):
    sub_item_box = sub_item["region"] + [sub_item["rotation"]]
    ratios = [rotated_rect_utils.rotated_rect_contains_ratio(
        topic["region"] + [topic["rotation"]], sub_item_box) for topic in topic_items]

    ret = []
    for topic, ratio in zip(topic_items, ratios):
        if ratio > thresh:
            ret.append(topic)
    return ret


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


def put_queue(progress_queue, text1, text2, path, tokenizer):
    source = {"image": path,
              "conversations":
                  [{"from": "human", "value": text1},
                   {"from": "gpt", "value": text2}]}
    input_id = preprocess(source, tokenizer)
    if len(input_id) > 2000:
        progress_queue.put(('toolong', path))
    else:
        progress_queue.put(('success', source))


def get_box(item, w, h):
    points = rotated_rect_utils.lefttop_rightbottom_theta_to_4points([*item['region'], item['rotation']])
    x0 = int(min(point[0] for point in points) * 1000 / w)
    y0 = int(min(point[1] for point in points) * 1000 / h)
    x1 = int(max(point[0] for point in points) * 1000 / w)
    y1 = int(max(point[1] for point in points) * 1000 / h)
    box = f'<box>({x0},{y0}),({x1},{y1})</box>'
    return box


def process(files, dir, tokenizer, progress_queue):
    for path in files:
        json_path = (dir+path)[:-4] + '.json'
        if not os.path.exists(json_path):
            json_path = (dir+path)[:-4] + '.txt'
            if not os.path.exists(json_path):
                print('not exists:', json_path)
                progress_queue.put(('error', json_path))
                continue
        try:
            data = json.load(open(json_path, encoding='utf-8'))
            skip = None
            items = []
            for item in data['regions']:
                if item['cls'] not in [1, 3, 10]:
                    continue
                if abs(item['rotation']) > 3:
                    skip = 'rotation'
                    break
                if item['cls'] != 3:
                    text = item['result'][0]
                    if text == '':
                        continue
                    if text.count('@') > 1:
                        skip = '@@@'
                        break
                items.append(item)
            if skip:
                progress_queue.put((skip, path))
                continue
            if len(items) < 10:
                progress_queue.put(('line', path))
                continue
            img = cv2.imread(dir+path)
            h, w = img.shape[:2]
            text2 = ''
            items = sorted(items, key=lambda x: x['region'][0])
            items = sorted(items, key=lambda x: x['region'][1])
            for item in items:
                if item['cls'] == 3:
                    text = '<pic>'
                else:
                    text = item['result'][0]
                box = get_box(item, w, h)
                text2 += f'<ref>{text}</ref>{box}'
            put_queue(progress_queue, '<image>', text2, path, tokenizer)
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue


def process_parallel(files, dir, tokenizer, progress_queue, output_path):
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, dir, tokenizer, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'rotation': 0, '@@@': 0, 'line': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'],
                                     toolong=counts['toolong'], rotation=counts['rotation'],
                                     aaa=counts['@@@'], line=counts['line'])
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


def prepare_det(tokenizer, parallel):
    with open('/mnt/ceph2/datasets/tiku/pretrain/files_det.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    # files = files[:128]
    progress_queue = multiprocessing.Queue()
    dir = '/mnt/ceph2/datasets/YYT_DET_20210602/'
    if parallel:
        process_parallel(files, dir, tokenizer, progress_queue, '/mnt/ceph2/datasets/tiku/pretrain/conversations_box_det.json')
    else:
        process(files, dir, tokenizer, progress_queue)


def prepare_qms(tokenizer, parallel):
    with open('/mnt/ceph2/datasets/tiku/pretrain/files_qms.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    random.shuffle(files)
    # files = files[:50000]
    progress_queue = multiprocessing.Queue()
    dir = '/mnt/ceph2/datasets/qms/'
    if parallel:
        process_parallel(files, dir, tokenizer, progress_queue, '/mnt/ceph2/datasets/tiku/pretrain/conversations_box_qms.json')
    else:
        process(files, dir, tokenizer, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/facebook/opt-125m',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)
    tokenizer.add_tokens("</s>", special_tokens=True)
    tokenizer.add_tokens('<imgpad>', special_tokens=True)
    tokenizer.add_tokens('<img>', special_tokens=True)
    tokenizer.add_tokens('</img>', special_tokens=True)
    tokenizer.add_tokens('<box>', special_tokens=True)
    tokenizer.add_tokens('</box>', special_tokens=True)
    tokenizer.add_tokens('<ref>', special_tokens=True)
    tokenizer.add_tokens('</ref>', special_tokens=True)

    parallel = False if sys.gettrace() is not None else True
    # prepare_det(tokenizer, parallel)
    prepare_qms(tokenizer, parallel)


def check():
    import re
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/pretrain/conversations_box_qms.json', encoding='utf-8'))
    dir = '/mnt/ceph2/datasets/qms/'
    # dir = '/mnt/ceph2/datasets/YYT_DET_20210602/'
    # dir = '/mnt/ceph2/datasets/tiku/'
    print(len(conversations))
    # for c in tqdm(conversations):
    #     path = dir + c['image']
    #     if not os.path.exists(path):
    #         print(path)
    for c in tqdm(conversations):
        path = dir + c['image']
        # if '`' not in c['conversations'][1]['value']:
        #     continue
        print(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        for i in range(2):
            text = c['conversations'][i]['value']
            print(text)
            matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', text)
            for match in matches:
                x0, y0, x1, y1 = map(int, match)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    prepare()
