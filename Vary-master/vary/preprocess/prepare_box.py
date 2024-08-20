import os
import json
import re
import random
import sys
import cv2
import transformers
import multiprocessing
from convertor.docx2html import DocxConvertor
from convertor.common import get_regions, match_pics
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
        if not os.path.exists(doc_path) or not os.path.exists(path):
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
            if any(font in text2 for font in ['方正姚体', '华文新魏', '华文隶书']):
                progress_queue.put(('font', path))
                continue
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
                text2 = text2.replace(match[0], f'<ref><pic></ref><box>({x1},{y1}),({x2},{y2})</box>')
            if skip:
                progress_queue.put(('toosmall', path))
                continue
            if with_font:
                text1 = f'<image>\nConvert with font:'
            else:
                text1 = f'<image>\nConvert:'
            put_queue(progress_queue, path, text1, text2, tokenizer)
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue


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
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'unmatch': 0, 'toomany': 0, 'font': 0, '□': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'],
                                     unmatch=counts['unmatch'], toomany=counts['toomany'], font=counts['font'],
                                     o=counts['□'])
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
    random.seed(0)
    files1 = glob('/mnt/ceph2/datasets/tiku7/words*/**/*.docx', recursive=True)
    # random.shuffle(files1)
    # files1 = files1[:1004]

    files2 = glob('/mnt/ceph2/datasets/tiku6/words_s/**/*.docx', recursive=True)
    files = glob('/mnt/ceph2/datasets/tiku6/words1/**/*.docx', recursive=True)
    # random.shuffle(files)
    # files2 += files[:1022]
    files2 += files

    files = glob('/mnt/ceph2/datasets/tiku4/words0-2/**/*.docx', recursive=True)
    random.shuffle(files)
    files3 = []
    for path in files:
        name = os.path.basename(path)[:-5]
        if name[-4] == '-' and name[-2] == '0' and name[-1] == '2':
            files3.append(path)
            # if len(files3) >= 1218:
            #     break

    files4 = glob('/mnt/ceph2/datasets/tiku5/words0/**/*.docx', recursive=True)
    files4 += glob('/mnt/ceph2/datasets/tiku5/words0-2/**/*.docx', recursive=True)
    # random.shuffle(files4)
    # files4 = files4[:1091]

    files = files1 + files2 + files3 + files4
    print(len(files))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_box2.json')
    else:
        process(files, tokenizer, True, True, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph2/datasets/tiku/sjb/0322/*.docx', recursive=True)
    files += glob(f'/mnt/ceph2/datasets/tiku/sjb/0401/*.docx', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_sjb_box.json')
    else:
        process(files, tokenizer, True, False, progress_queue)


def process_camera(files, tokenizer, footer, with_font, progress_queue):
    docx = DocxConvertor(with_font=with_font)
    for path in files:
        doc_path = path.replace('_pp_1213', '')[:-6] + '.docx'
        if not os.path.exists(doc_path):
            doc_path = path.replace('_pp_1213', '')[:-5] + '.docx'
            if not os.path.exists(doc_path):
                print('not exists:', doc_path)
                progress_queue.put(('error', doc_path))
                continue
        det_path = path[:-4] + '_det.json'
        if not os.path.exists(det_path):
            print('not exists:', det_path)
            progress_queue.put(('error', det_path))
            continue
        box_path = None
        try:
            img = cv2.imread(path)
            h, w = img.shape[:2]
            data = json.load(open(det_path, encoding='utf-8'))
            regions, pics, tables = get_regions(data, w, h)
            ocr = ''
            for region in regions:
                ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
            text2 = docx.docx2html(doc_path, format=2, box_path=box_path, footer=footer)
            # if text2.count('<img>') > 4:
            #     progress_queue.put(('toomany', path))
            #     continue
            if text2.count('<img>') != len(pics):
                progress_queue.put(('unmatch', path))
                continue
            pics = sorted(pics, key=lambda x: x['bbox'][0])
            pics = sorted(pics, key=lambda x: x['bbox'][1])
            for pic in pics:
                i = text2.find('<img>')
                text2 = text2[:i] + pic['result'] + text2[i + 5:]
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        if with_font:
            text1 = f'<image>\nOCR:\n{ocr}\nConvert with font:'
        else:
            text1 = f'<image>\nOCR:\n{ocr}\nConvert:'
        put_queue(progress_queue, path, text1, text2, tokenizer)


def prepare_camera(tokenizer, parallel):
    files = glob('/mnt/ceph2/datasets/tiku5/words_c/**/*.png', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku5/words_c/**/*.jpg', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process_camera, files, tokenizer, True, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_camera_det14.json')
    else:
        process_camera(files, tokenizer, True, True, progress_queue)


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
    prepare_tiku(tokenizer, parallel)
    # prepare_sjb(tokenizer, parallel)
    # prepare_camera(tokenizer, parallel)


def check():
    from convertor.docx2html import DocxConvertor
    docx = DocxConvertor()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_box2.json', encoding='utf-8'))
    # for c in conversations:
    #     if 'grapefruit. Lesson 36' in c['image']:
    #         print(c['image'])
    #         break
    random.shuffle(conversations)
    for c in tqdm(conversations):
        path = '/mnt/ceph2/datasets/' + c['image']
        text1 = c['conversations'][0]['value']
        text2 = c['conversations'][1]['value']
        print(path)
        print(text1)
        print(text2)
        text2 = text2.replace('<ref><pic></ref>', '')
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
