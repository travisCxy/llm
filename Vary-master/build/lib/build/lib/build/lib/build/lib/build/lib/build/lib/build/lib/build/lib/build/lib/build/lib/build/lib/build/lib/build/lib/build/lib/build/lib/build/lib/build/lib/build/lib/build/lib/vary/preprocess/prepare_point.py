import os
import json
import re
import random
import sys
import transformers
import multiprocessing
from docx import Docx
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


def process(files, tokenizer, footer, progress_queue):
    docx = Docx()
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
                x1 = int(float(match[1]) * 1000)
                y1 = int(float(match[2]) * 1000)
                x2 = int((float(match[1]) + float(match[3])) * 1000)
                y2 = int((float(match[2]) + float(match[4])) * 1000)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                text2 = text2.replace(match[0], f'<point>({center_x},{center_y})</point>')
            if skip:
                progress_queue.put(('toosmall', path))
                continue
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        text1 = '<image>\nConvert the document to html/latex formart:'
        source = {"image": path.replace('/mnt/ceph2/datasets/tiku/', ''),
                  "conversations":
                      [{"from": "human", "value": text1},
                       {"from": "gpt", "value": text2}]}
        input_id = preprocess(source, tokenizer)
        if len(input_id) > 2048:
            progress_queue.put(('toolong', path))
            continue
        progress_queue.put(('success', source))

        suffixes = ['_crop_hw_1226', '_crop_hwn_1226',
                    '_crop_pp_1213', '_crop_ppn_1213',
                    '_crop_ppc_1225', '_crop_ppcn_1225']
        if any(s in path for s in suffixes):
            match = re.search(r'_crop_[a-z]+?_\d+?\.', path)
            suffix = match.group()[:-1]
            if '<point>' in text2 and '_hw_' in suffix:
                continue
            suffixes.remove(suffix)
            for s in suffixes:
                if '<point>' in text2 and (s == '_crop_hw_1226' or 'n' in s):
                    continue
                new_path = path.replace(suffix, s)
                source = {"image": new_path.replace('/mnt/ceph2/datasets/tiku/', ''),
                          "conversations":
                              [{"from": "human", "value": text1},
                               {"from": "gpt", "value": text2}]}
                progress_queue.put(('success', source))


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
    files = glob(f'/mnt/ceph2/datasets/tiku/images0/**/*.jpg', recursive=True)
    random.shuffle(files)
    files0 = files[:10000]

    files = glob(f'/mnt/ceph2/datasets/tiku/images0-2/**/*.jpg', recursive=True)
    random.shuffle(files)
    files02 = []
    for path in files:
        name = os.path.basename(path)[:-4]
        if name[-4] != '-':
            continue
        if name[-2] == '0' and name[-1] == '0':
            files02.append(path)
            if len(files02) >= 10000:
                break
    for path in files:
        name = os.path.basename(path)[:-4]
        if name[-4] != '-':
            continue
        if name[-2] == '0' and name[-1] == '2':
            files02.append(path)
            if len(files02) >= 40000:
                break

    with open('/mnt/ceph2/datasets/tiku/images_split/files.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    files = [f'/mnt/ceph2/datasets/tiku/images_split/{file}' for file in files if file]
    random.shuffle(files)
    files = files[:30000]

    files = files0 + files02 + files

    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_point.json')
    else:
        process(files, tokenizer, False, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph2/datasets/tiku/sjb/0322/*.docx', recursive=True)
    files += glob(f'/mnt/ceph2/datasets/tiku/sjb/0401/*.docx', recursive=True)
    # files = ['/mnt/ceph2/datasets/tiku/sjb/0322/200888461-1_crop_ppc_1225.docx']
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_sjb_point.json')
    else:
        process(files, tokenizer, True, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    prepare_tiku(tokenizer, parallel)
    # prepare_sjb(tokenizer, parallel)


def check():
    import cv2
    docx = Docx()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_point.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        path = '/mnt/ceph2/datasets/tiku/' + c['image']
        print(path)
        print(c['conversations'][0]['value'])
        text = c['conversations'][1]['value']
        print(text)
        pretty_html = docx.pretty(text, 2, path)
        with open(r'/mnt/ceph2/datasets/tiku/vary/test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))

        matches = re.findall(r'(<point>\((\d+),(\d+)\)<\/point>)', text)
        for match in matches:
            x = int(match[1])
            y = int(match[2])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()
