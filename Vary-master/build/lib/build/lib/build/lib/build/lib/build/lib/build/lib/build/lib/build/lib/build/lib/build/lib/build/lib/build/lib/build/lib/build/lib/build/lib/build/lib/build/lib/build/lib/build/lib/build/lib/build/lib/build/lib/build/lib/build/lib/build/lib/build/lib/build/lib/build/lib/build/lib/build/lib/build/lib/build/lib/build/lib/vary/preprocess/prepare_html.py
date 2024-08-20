import os
import json
import re
import random
import sys
import shutil
import transformers
import multiprocessing
from docx import Docx
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


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


def process(files, tokenizer, footer, progress_queue):
    docx = Docx()
    # template = json.load(open(f'template_html.json', encoding='utf-8'))
    for path in files:
        if path.endswith('.docx'):
            doc_path = path
            path = path.replace('/words', '/images')[:-5] + '.jpg'
        else:
            doc_path = path.replace('/images', '/words')[:-4] + '.docx'
        if not os.path.exists(doc_path):
            print('not exists:', doc_path)
            progress_queue.put(('error', doc_path))
            continue
        box_path = path[:-4] + '.json'
        if not os.path.exists(box_path):
            box_path = doc_path[:-5] + '.json'
            if not os.path.exists(box_path):
                box_path = None
        try:
            text2 = docx.docx2html(doc_path, format=0, box_path=box_path, footer=footer)
            matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
            skip = False
            for match in matches:
                if float(match[3]) < 0.05:
                    skip = True
                    break
                x1 = int(float(match[1]) * 1000)
                y1 = int(float(match[2]) * 1000)
                x2 = int((float(match[1]) + float(match[3])) * 1000)
                y2 = int((float(match[2]) + float(match[4])) * 1000)
                text2 = text2.replace(match[0], f'<box>({x1},{y1}),({x2},{y2})</box>')
            if skip:
                progress_queue.put(('toosmall', path))
                continue
            # text1 = '<image>\nConvert the document image to html/latex formart:'
            text1 = '<image>\nThe OCR result of the image:'
            source = {"image": path.replace('/mnt/ceph2/datasets/tiku/', ''),
                      "conversations":
                          [{"from": "human", "value": text1},
                           {"from": "gpt", "value": text2}]}
            input_id = preprocess(source, tokenizer)
            if len(input_id) > 2048:
                progress_queue.put(('toolong', path))
                continue
            progress_queue.put(('success', source))
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue


def process_parallel(files, tokenizer, footer, progress_queue, output_path):
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, tokenizer, footer, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'toosmall': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'],
                                     toosmall=counts['toosmall'])
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
    files0 = glob(f'/mnt/ceph2/datasets/tiku/images0/**/*.jpg', recursive=True)
    files0 += glob(f'/mnt/ceph2/datasets/tiku/images1/**/*.jpg', recursive=True)
    files0 += glob(f'/mnt/ceph2/datasets/tiku/images2/**/*.jpg', recursive=True)

    files = glob(f'/mnt/ceph2/datasets/tiku/images0-2/**/*.jpg', recursive=True)
    random.shuffle(files)
    files02 = []
    for path in files:
        name = os.path.basename(path)[:-4]
        if name[-4] != '-':
            continue
        files02.append(path)

    with open('/mnt/ceph2/datasets/tiku/images_split/files.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    files = [f'/mnt/ceph2/datasets/tiku/images_split/{file}' for file in files if file]
    random.shuffle(files)
    files = files[:1250000-len(files0)-len(files02)]

    files = files0 + files02 + files

    progress_queue = multiprocessing.Queue()
    if parallel:
        # process_parallel(files, tokenizer, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_html_1M.json')
        process_parallel(files, tokenizer, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_ocr_1M.json')
    else:
        process(files, tokenizer, False, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    prepare_tiku(tokenizer, parallel)


def check():
    import re
    import cv2
    docx = Docx()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_html_1M.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        text = c['conversations'][1]['value']
        path = '/mnt/ceph/tiku/' + c['image']
        print(path)
        print(c['conversations'][0]['value'])
        print(text)
        pretty_html = docx.pretty(text, 2, path)
        with open(r'/mnt/ceph2/datasets/tiku/vary/test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        # matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', c['conversations'][1]['value'])
        # for match in matches:
        #     x0, y0, x1, y1 = map(int, match)
        #     cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    # prepare()

    print(len(glob(f'/mnt/ceph2/datasets/tiku/images0/**/*.jpg', recursive=True)))
    print(len(glob(f'/mnt/ceph2/datasets/tiku/images1/**/*.jpg', recursive=True)))
    print(len(glob(f'/mnt/ceph2/datasets/tiku/images2/**/*.jpg', recursive=True)))
    print(len(glob(f'/mnt/ceph2/datasets/tiku/images0-2/**/*.jpg', recursive=True)))
    with open('/mnt/ceph2/datasets/tiku/words_split/files.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    print(len(files))
