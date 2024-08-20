import os
import json
import re
import random
import shutil
import sys
import cv2
import transformers
import multiprocessing
from convertor.docx2html import DocxConvertor
from convertor.docx2docx import create_clos_for_pic
from convertor.docx2img import docx2img
from convertor.common import get_regions, match_pics
from glob import glob
from tqdm import tqdm


def get_docx():
    files = glob('/mnt/ceph2/datasets/tiku/words2/**/*.docx', recursive=True)
    # docx = DocxConvertor(with_font=False)
    count = 0
    with open('/mnt/ceph2/datasets/tiku6/files2.txt', 'w', encoding='utf-8') as f:
        for path in tqdm(files):
            if '数学' not in path:
                continue
            # try:
            #     text = docx.docx2html(path, format=2)
            # except Exception as e:
            #     continue
            # if text.count('\sqrt') > 2:
            f.write(path + '\n')
            count += 1
            if count >= 30000:
                break
    print(count)


def prepare_imgs():
    files = glob('/mnt/ceph2/datasets/tiku6/words2/**/*.docx', recursive=True)
    for path in files:
        dir = os.path.dirname(path).replace('words', 'images')
        if not os.path.exists(dir):
            os.makedirs(dir)
    docx2img(files)


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
    source = {"image": path.replace('/mnt/ceph2/datasets/', ''),
              "conversations":
                  [{"from": "human", "value": text1},
                   {"from": "gpt", "value": text2}]}
    input_id = preprocess(source, tokenizer)
    if len(input_id) > 3600:
        queue.put(('toolong', path))
    else:
        queue.put(('success', source))


def process(files, tokenizer, footer, with_font, progress_queue):
    docx = DocxConvertor(with_font=with_font)
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
            text2 = docx.docx2html(doc_path, format=2, box_path=box_path, footer=footer)
            matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
            skip = False
            for match in matches:
                if float(match[3]) < 0.05:
                    skip = True
                    break
                text2 = text2.replace(match[0], '<pic>')
            if skip:
                progress_queue.put(('toosmall', path))
                continue
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        if with_font:
            text1 = f'<image>\nConvert with font:'
        else:
            text1 = f'<image>\nConvert:'
        put_queue(progress_queue, path, text1, text2, tokenizer)


def process_parallel(target, files, tokenizer, footer, with_font, progress_queue, output_path):
    chunk_size = len(files) // multiprocessing.cpu_count()
    if chunk_size == 0:
        chunk_size = len(files)
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=target, args=(chunk, tokenizer, footer, with_font, progress_queue)) for chunk in
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
    files = glob('/mnt/ceph2/datasets/tiku/words2/**/*.docx', recursive=True)
    files = [f for f in files if '数学' in f]
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det19.json')
    else:
        process(files, tokenizer, True, True, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    prepare_tiku(tokenizer, parallel)


def check():
    from convertor.docx2html import DocxConvertor
    docx = DocxConvertor()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det19.json', encoding='utf-8'))
    # count = 0
    # for c in tqdm(conversations):
    #     path = '/mnt/ceph2/datasets/' + c['image']
    #     if '一年级' not in path and '二年级' not in path:
    #         continue
    #     count += 1
    # print(count)
    print(len(conversations))
    random.shuffle(conversations)
    for c in tqdm(conversations):
        path = '/mnt/ceph2/datasets/' + c['image']
        # if 'tiku7' not in path:
        #     continue
        # if '/images_s/' not in path:
        #     continue
        # if '2019-2020学年人教版三年级上册期中考试数学试卷2_1077697_数学_3' not in path:
        #     continue
        text1 = c['conversations'][0]['value']
        text2 = c['conversations'][1]['value']
        # if text.count('<footer') < 1:
        #     continue
        # if text.count('\sqrt') < 3:
        #     continue
        # if c['conversations'][0]['value'].count('□') != 2:
        #     continue
        # if all(t not in text2 for t in ['计算题', '竖式计算', '直接写出得数', '用递等式计算']):
        #     continue
        # if all(t not in text2 for t in ['连线题']):
        #     continue
        # if 'img' not in text2:
        #     continue
        print(path)
        print(text1)
        print(text2)
        print(len(text2))
        # with open(r'output/ocr.txt', 'w', encoding='utf-8') as f:
        #     f.write(c['conversations'][0]['value'])
        pretty_html = docx.pretty(text2, 2, path)
        with open(r'test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (768, 768))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    # prepare_imgs()
    # prepare()
    check()
