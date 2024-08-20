import os
import json
import re
import random
import sys
import cv2
import transformers
import multiprocessing
import numpy as np
from lxml import html
from tqdm import tqdm


with open('/mnt/ceph2/datasets/im2latex/im2latex_formulas.lst', 'r', encoding='Windows-1252', newline='\n') as f:
    formulas = [line.strip() for line in f.readlines()]


def calc_margin(is_white):
    margin = 0
    for row in is_white:
        if row.all():
            margin += 1
        else:
            break
    return margin


def get_margin(img):
    h, w = img.shape[:2]
    is_white = np.all(img >= 250, axis=-1)
    top = calc_margin(is_white) - 20
    bottom = h - calc_margin(reversed(is_white)) + 20
    left = calc_margin(is_white.T) - 20
    right = w - calc_margin(reversed(is_white.T)) + 20
    return top, bottom, left, right


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
    labels = ['\\vspace', '\\hspace', '\\label', '\\dag', '\\sp', '\\sb',
              '\\begin{picture}', '\\unitlength', '\\P']
    pattern = re.compile(r'\\left(.*?)\\right', re.DOTALL)
    name = ''
    for data in dataset:
        data = data.strip().split()
        path = os.path.join('/mnt/ceph2/datasets/im2latex/formula_images', data[1] + '.png')
        try:
            formula = formulas[int(data[0])]
            if '\\label' in formula:
                formula = re.sub(r'\\label\{.*?\}', '', formula)
            if any(label in formula for label in labels):
                progress_queue.put(('label', path))
                continue
            if '\\\\' in pattern.sub('', formula):
                progress_queue.put(('skip', path))
                continue
            img = cv2.imread(path)
            top, bottom, left, right = get_margin(img)
            img = img[top:bottom, left:right]
            h, w = img.shape[:2]
            if w > canvas_width:
                img = cv2.resize(img, (canvas_width, int(h * canvas_width / w)))
                h, w = img.shape[:2]
            if y_offset + h > canvas_height or len(body) + len(formula) > 2000:
                cv2.imwrite(f'/mnt/ceph2/datasets/im2latex/images/{name}.jpg', canvas)
                put_queue(progress_queue, f'{name}.jpg', f'<image>\nConvert:', f'<body>{body}</body>', tokenizer)
                canvas, canvas_width, canvas_height = create_canvas()
                y_offset = 0
                body = ''
                name = ''
            else:
                progress_queue.put(('formula', data[1]))
            if w > canvas_width:
                img = cv2.resize(img, (canvas_width, int(h * canvas_width / w)))
                h, w = img.shape[:2]
            canvas[y_offset:y_offset + h, :w] = img
            y_offset += h
            body += f'<p>\\({formula}\\)</p>'
            name += data[0]
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
    with open('/mnt/ceph2/datasets/im2latex/im2latex_train.lst', 'r') as f:
        dataset = f.readlines()
    random.seed(0)
    random.shuffle(dataset)
    if parallel:
        process_parallel(process, dataset, tokenizer, progress_queue, '/mnt/ceph2/datasets/im2latex/conversations_latex.json')
    else:
        process(dataset, tokenizer, progress_queue)


def check():
    conversations = json.load(open('/mnt/ceph2/datasets/im2latex/conversations_latex.json', encoding='utf-8'))
    random.shuffle(conversations)
    for c in tqdm(conversations):
        path = c['image']
        text = c['conversations'][1]['value']
        print(path)
        print(c['conversations'][0]['value'])
        print(text)
        str = render(text)
        img = cv2.imread(f'/mnt/ceph2/datasets/im2latex/images/' + path)
        img = cv2.resize(img, (800, 800))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    # # with open("/mnt/server_data2/mathlens/mathlens_train.txt", 'r') as f:
    # with open("/mnt/server_data2/mathlens/mathlens_train_gen.txt", 'r') as f:
    # # with open("/mnt/server_data2/mathlens/mathlens_train_aizuoye.txt", 'r') as f:
    #     for line in f:
    #         data = line.strip().split('\t')
    #         text = data[1]
    #         if text.count('\sqrt') < 3:
    #             continue
    #         path = os.path.join('/mnt/server_data2/mathlens', data[0])
    #         print(path)
    #         print(text)
    #         img = cv2.imread(path)
    #         cv2.imshow('img', img)
    #         if cv2.waitKey(0) == 27:
    #             break

    # import chardet
    # with open('/mnt/ceph2/datasets/im2latex/im2latex_formulas.lst', 'rb') as f:
    #     result = chardet.detect(f.read())
    #
    # encoding = result['encoding']
    # print(f"Detected encoding: {encoding}")

    # check()

    # with open('/mnt/ceph2/datasets/im2latex/im2latex_train.lst', 'r') as f:
    #     dataset = f.readlines()
    # labels = ['\\vspace', '\\hspace']
    # pattern = re.compile(r'\\left(.*?)\\right', re.DOTALL)
    # name = ''
    # counts = [0] * 4
    # for data in dataset:
    #     data = data.strip().split()
    #     path = os.path.join('/mnt/ceph2/datasets/im2latex/formula_images', data[1] + '.png')
    #     formula = formulas[int(data[0])]
    #     if '\\label' not in formula:
    #         counts[0] += 1
    #         continue
    #     if any(label in formula for label in labels):
    #         counts[1] += 1
    #         continue
    #     if '\\\\' in pattern.sub('', formula):
    #         counts[2] += 1
    #         continue
    #     counts[3] += 1
    #     formula2 = re.sub(r'\\label\{.*?\}', '', formula)
    #     print(formula)
    #     print(formula2)
    #     render(f'<p>\\({formula}\\)</p><p>\\({formula2}\\)</p>')
    #     img = cv2.imread(path)
    #     top, bottom, left, right = get_margin(img)
    #     img = img[top:bottom, left:right]
    #     cv2.imshow('img', img)
    #     if cv2.waitKey(0) == 27:
    #         break
    # print(counts)

    import base64
    with open(r'/mnt/ceph2/pretrained/Ucas/vary-llava80k/qwen.tiktoken', 'rb') as f:
        contents = f.read()
    tokens = set()
    for line in tqdm(contents.splitlines()):
        try:
            token = base64.b64decode(line.split()[0]).decode('utf-8')
            tokens.add(token)
        except:
            pass
    print(len(tokens))

    for c in ['🍎', '🚲', '📚', '🎸', '🏠']:
        if c not in tokens:
            print(c)


    # with open(r'/mnt/ceph2/datasets/printed_text/synset.txt', 'r') as f:
    #     synsets = [line.strip() for line in f.readlines()]
    # synsets = set(synsets)
    # print(len(synsets))
    #
    # chars = []
    # for synset in synsets:
    #     if synset not in tokens:
    #         chars.append(synset)
    # print(len(chars))
    # print(chars)

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
    #
    # with open(r'/mnt/ceph2/datasets/printed_text/bak_store_filtered_6.txt', 'r') as f:
    #     chars = set()
    #     for line in tqdm(f):
    #         data = line.strip().split('\t')
    #         text = data[1]
    #         if all(char in tokens for char in text) and any(char not in train_chars for char in text):
    #             skip = True
    #             for char in text:
    #                 if char not in train_chars:
    #                     if char not in chars:
    #                         chars.add(char)
    #                         skip = False
    #             if skip:
    #                 continue
    #             print(text, [char for char in text if char not in train_chars])
    #             img = cv2.imread(f'/mnt/ceph2/datasets/printed_text/{data[0]}')
    #             cv2.imshow('img', img)
    #             if cv2.waitKey(0) == 27:
    #                 break
