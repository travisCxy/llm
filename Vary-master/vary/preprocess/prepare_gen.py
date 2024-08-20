import os
import json
import re
import random
import shutil
import uuid
import sys
import cv2
import transformers
import multiprocessing
import numpy as np
from lxml import html
from tqdm import tqdm


def render(body):
    str = '''<!DOCTYPE html>
            <html>
            <head>
                <script type="text/javascript" async
                    src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
                </script>
                <style>
                    body {
                        margin: 50px;
                    }
                    p {
                        white-space: pre-wrap;
                        text-align: justify;
                    }
                </style>
            </head>
            ''' + body + '''</html>'''
    root = html.fromstring(str)
    pretty_html = html.tostring(root, encoding='unicode', pretty_print=True)
    with open(r'test.html', 'w', encoding='utf-8') as f2:
        f2.write(pretty_html)
    return str


def create_canvas():
    canvas_width = random.randint(512, 2048)
    canvas_height = int(canvas_width * random.uniform(0.5, 2))
    # print(canvas_width, canvas_height)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    return canvas, canvas_width, canvas_height


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


def put_queue(queue, path, text1, text2, tokenizer):
    source = {"image": path,
              "conversations":
                  [{"from": "human", "value": text1},
                   {"from": "gpt", "value": text2}]}
    input_id = preprocess(source, tokenizer)
    if len(input_id) > 3600:
        queue.put(('toolong', path))
    else:
        queue.put(('success', source))


def process(dataset, tokenizer, progress_queue):
    canvas, canvas_width, canvas_height = create_canvas()
    y_offset = 0
    body = ''
    for data in dataset:
        path, text = data
        try:
            img = cv2.imread(path)
            h, w = img.shape[:2]
            if w > canvas_width - 50:
                img = cv2.resize(img, (canvas_width - 50, int(h * (canvas_width - 50) / w)))
                h, w = img.shape[:2]
            if y_offset + h + 50 > canvas_height or len(body) + len(text) > 2000:
                name = uuid.uuid4().hex
                path = f'/mnt/ceph2/datasets/gen/images2/{name}.jpg'
                if os.path.exists(path):
                    print('exists:', path)
                cv2.imwrite(path, canvas)
                # img = cv2.imread(f'/mnt/ceph2/datasets/gen/images2/{name}.jpg')
                # cv2.imshow('img', img)
                # cv2.waitKey()
                put_queue(progress_queue, f'{name}.jpg', f'<image>\nConvert:', f'<body>{body}</body>', tokenizer)
                canvas, canvas_width, canvas_height = create_canvas()
                y_offset = 0
                body = ''
            else:
                progress_queue.put(('formula', data[1]))
            if w > canvas_width - 50:
                img = cv2.resize(img, (canvas_width - 50, int(h * (canvas_width - 50) / w)))
                h, w = img.shape[:2]
            canvas[y_offset:y_offset + h, 50:w+50] = img
            y_offset += h + 50
            body += f'<p>{text}</p>'
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue


def process_parallel(target, dataset, tokenizer, progress_queue, output_path):
    chunk_size = len(dataset) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    dataset_chunks = [dataset[i:i + chunk_size] for i in range(0, len(dataset), chunk_size)]
    runners = [multiprocessing.Process(target=target, args=(chunk, tokenizer, progress_queue)) for chunk in dataset_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(dataset))
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'skip': 0, 'formula': 0, 'label': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'],
                                     skip=counts['skip'], formula=counts['formula'], label=counts['label'])
        except Exception as e:
            if all(not runner.is_alive() for runner in runners):
                break
            continue
        progress_bar.update(1)
    progress_bar.close()
    print(len(conversations), counts)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    progress_queue = multiprocessing.Queue()

    dataset = []

    # lines = []
    # with open("/mnt/server_data2/mathlens/mathlens_train.txt", 'r') as f:
    #     lines += [line for line in f.readlines() if line.count('\sqrt') > 0]
    # with open("/mnt/server_data2/mathlens/mathlens_train_gen.txt", 'r') as f:
    #     lines += [line for line in f.readlines() if line.count('\sqrt') > 0]
    # # print(len(lines))
    #
    # for line in tqdm(lines):
    #     path, text = line.strip().split('\t')
    #     path2 = os.path.join('/mnt/ceph2/datasets/gen', path)
    #     # if not os.path.exists(path2):
    #     #     shutil.copyfile(os.path.join('/mnt/server_data2/mathlens', path), path2)
    #     dataset.append((path2, f'\\({text.replace(" ", "")}\\)'))

    # import base64
    # with open(r'/mnt/ceph2/pretrained/Ucas/vary-llava80k/qwen.tiktoken', 'rb') as f:
    #     contents = f.read()
    # tokens = set()
    # for line in tqdm(contents.splitlines()):
    #     try:
    #         token = base64.b64decode(line.split()[0]).decode('utf-8')
    #         tokens.add(token)
    #     except:
    #         pass
    # print(len(tokens))

    # from collections import Counter
    # conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det14.json', encoding='utf-8'))
    # conversations += json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det14.json', encoding='utf-8'))
    # conversations += json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_camera_det14.json', encoding='utf-8'))
    # char_counter = Counter()
    # for c in conversations:
    #     text = c['conversations'][1]['value']
    #     char_counter.update(text)
    # print(len(char_counter))
    # print(char_counter)

    # train_chars = set(char_counter.keys())
    # with open('/mnt/ceph2/datasets/gen/bak_store_filtered.txt', 'w') as f2:
    #     for i in range(8):
    #         path = f'/mnt/ceph2/datasets/printed_text/bak_store_filtered_{i}.txt'
    #         with open(path, 'r') as f:
    #             for line in tqdm(f):
    #                 d = line.strip().split('\t')
    #                 text = d[1]
    #                 if all(char in tokens for char in text) and any(char not in train_chars for char in text):
    #                     path = os.path.join('/mnt/ceph2/datasets/printed_text', d[0])
    #                     dataset.append((path, text))
    #                     f2.write(f'{path}\t{text}\n')
    with open('/mnt/ceph2/datasets/gen/bak_store_filtered.txt', 'r') as f:
        for line in f:
            path, text = line.strip().split('\t')
            dataset.append((path, text))
    print(len(dataset))

    random.shuffle(dataset)

    if parallel:
        process_parallel(process, dataset, tokenizer, progress_queue, '/mnt/ceph2/datasets/gen/conversations.json')
    else:
        process(dataset, tokenizer, progress_queue)


def check():
    conversations = json.load(open('/mnt/ceph2/datasets/gen/conversations2.json', encoding='utf-8'))
    random.shuffle(conversations)
    for c in tqdm(conversations):
        path = c['image']
        text = c['conversations'][1]['value']
        print(path)
        print(c['conversations'][0]['value'])
        print(text)
        str = render(text)
        img = cv2.imread(f'/mnt/ceph2/datasets/gen/images2/' + path)
        img = cv2.resize(img, (800, 800))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    # prepare()
    check()

    # conversations = json.load(open('/mnt/ceph2/datasets/gen/conversations.json', encoding='utf-8'))
    # for c in tqdm(conversations):
    #     c['image'] = c['image'][:-4] + '_pp_1213.jpg'
    #     assert os.path.exists(f'/mnt/ceph2/datasets/gen/images2/' + c['image'])
    # with open('/mnt/ceph2/datasets/gen/conversations2.json', 'w', encoding='utf-8') as f:
    #     json.dump(conversations, f, ensure_ascii=False, indent=2)
