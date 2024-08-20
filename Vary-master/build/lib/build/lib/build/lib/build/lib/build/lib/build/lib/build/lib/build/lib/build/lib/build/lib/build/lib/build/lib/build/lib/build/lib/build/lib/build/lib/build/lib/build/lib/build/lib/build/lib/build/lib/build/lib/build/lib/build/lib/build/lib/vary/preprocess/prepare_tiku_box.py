import os
# import fitz
import random
import json
import re
import multiprocessing
import transformers
from docx import Docx
from glob import glob
from tqdm import tqdm


def process_box(chunk, progress_queue):
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    template = json.load(open(f'template_box.json', encoding='utf-8'))
    for item in chunk:
        text1 = random.choice(template)
        text2 = f'{len(item["boxes"])} illustrations: '
        for box in item["boxes"]:
            text2 += f'@[{int(box[0]*1024)}, {int(box[1]*1024)}, {int(box[2]*1024)}, {int(box[3]*1024)}]@ '
        text2 = text2[:-1] + '.'
        text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
        tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
        if len(tokenized.input_ids[0]) > 2048:
            progress_queue.put(('toolong', None))
            continue
        progress_queue.put(('success', {"image": item["image"],
                                        "conversations":
                                        [{"from": "human", "value": text1},
                                         {"from": "gpt", "value": text2}]}))


def process_next_box(chunk, progress_queue):
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    template = json.load(open(f'template_next_box.json', encoding='utf-8'))
    for item in chunk:
        text1 = random.choice(template)
        text2 = f'{len(item["boxes"])} illustrations: '
        boxes = []
        for box in item["boxes"]:
            text2 += '<at> <boxes>, '
            boxes.append((box[0], box[1], box[2], box[3]))
        text2 = text2[:-2] + '.'
        text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
        tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
        if len(tokenized.input_ids[0]) > 2048:
            progress_queue.put(('toolong', None))
            continue
        progress_queue.put(('success', {"image": item["image"],
                                        "boxes": boxes,
                                        "norm": True,
                                        "conversations":
                                        [{"from": "human", "value": text1},
                                         {"from": "gpt", "value": text2}]}))


def process_box2(chunk, progress_queue):
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    template = json.load(open(f'template3.json', encoding='utf-8'))
    for item in chunk:
        text1 = '<image>\n' + random.choice(template)
        text2 = item["conversations"][1]["value"]
        for box in item["boxes"]:
            box = f'<box>({int(box[0] * 1000)},{int(box[1] * 1000)}),({int(box[2] * 1000)},{int(box[3] * 1000)})</box>'
            k = text2.find('<ref><box>')
            text2 = text2[:k] + box + text2[k+10:]
        text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
        tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
        if len(tokenized.input_ids[0]) > 2048:
            progress_queue.put(('toolong', None))
            continue
        progress_queue.put(('success', {"image": item["image"],
                                        "conversations":
                                        [{"from": "human", "value": text1},
                                         {"from": "gpt", "value": text2}]}))


def process_html_box(chunk, progress_queue):
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    template = json.load(open(f'template_html.json', encoding='utf-8'))
    for item in chunk:
        text1 = '<image>\n' + random.choice(template)[:-1] + ' with illustrations. The coordinates of the illustrations must be ensured to be very precise. The coordinates are based on 1024x1024 image.'
        text2 = item["conversations"][1]["value"]
        for box in item["boxes"]:
            box = f'@[{int(box[0]*1024)}, {int(box[1]*1024)}, {int(box[2]*1024)}, {int(box[3]*1024)}]@'
            k = text2.find('<ref><box>')
            text2 = text2[:k] + box + text2[k+10:]
        text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
        tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
        if len(tokenized.input_ids[0]) > 2048:
            progress_queue.put(('toolong', None))
            continue
        progress_queue.put(('success', {"image": item["image"],
                                        "conversations":
                                        [{"from": "human", "value": text1},
                                         {"from": "gpt", "value": text2}]}))


def prepare():
    datas = json.load(open('/mnt/ceph/tiku/images_split/conversations_ocr_box_0.json', encoding='utf-8'))
    # progress_queue = multiprocessing.Queue()
    # process(datas, progress_queue)
    chunk_size = len(datas) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    data_chunks = [datas[i:i + chunk_size] for i in range(0, len(datas), chunk_size)]
    progress_queue = multiprocessing.Queue()
    runners = [multiprocessing.Process(target=process_box2, args=(chunk, progress_queue)) for chunk in data_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(datas))
    counts = {'success': 0, 'toolong': 0}
    conversations = []
    while not progress_bar.n >= len(datas):
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], toolong=counts['toolong'])
        except:
            if all(not runner.is_alive() for runner in runners):
                break
            continue
        progress_bar.update(1)
    progress_bar.close()
    if len(conversations) > 0:
        with open(f'/mnt/ceph/tiku/images_split/tiku_box2_0.json', 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    print(counts)


def check():
    import re
    import cv2
    docx = Docx()
    conversations = json.load(open('/mnt/ceph/tiku/images_split/tiku_box2_1.json', encoding='utf-8'))
    for c in tqdm(conversations):
        path = os.path.join('/mnt/ceph/tiku/images_split', c['image'])
        text = c['conversations'][1]['value']
        print(path)
        print(c['conversations'][0]['value'])
        print(text)
        # pretty_html = docx.pretty(text, 2, path)
        # with open(r'/mnt/ceph/Vary/test/test.html', 'w', encoding='utf-8') as f:
        #     f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', c['conversations'][1]['value'])
        for match in matches:
            x0, y0, x1, y1 = map(int, match)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # matches = re.findall(r'@\[.*?\]@', c['conversations'][1]['value'])
        # boxes = []
        # for match in matches:
        #     match = match[2:-2]
        #     boxes.append([float(x) / 1024 for x in match.split(', ')])
        # for box in boxes:
        #     x0, y0, x1, y1 = box
        #     x0, y0, x1, y1 = int(x0 * 768), int(y0 * 768), int(x1 * 768), int(y1 * 768)
        #     cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # for box in c['boxes']:
        #     x0, y0, x1, y1 = box
        #     x0, y0, x1, y1 = int(x0 * 768 / 1000), int(y0 * 768 / 1000), int(x1 * 768 / 1000), int(y1 * 768 / 1000)
        #     cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()
