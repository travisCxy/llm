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


def preprocess(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
):
    text = source["instruction"] + '\n' + source["input"] + '\n' + source["output"]
    tokenized = tokenizer(text, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
    return tokenized.input_ids[0]


def put_queue(queue, input, output, tokenizer):
    source = {"instruction": "根据OCR结果使用html/laetx格式还原文档。",
              "input": input,
              "output": output}
    input_id = preprocess(source, tokenizer)
    if len(input_id) > 4096:
        queue.put(('toolong', None))
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
        if '/sjb/' not in path and '/tiku6/' not in path and '/tiku7/' not in path:
            if os.path.exists(doc_path.replace('/tiku2', '/tiku5')):
                doc_path = doc_path.replace('/tiku2', '/tiku5')
                path = path.replace('/tiku2', '/tiku5')
            else:
                two = False
                if 'words0-2' in doc_path:
                    name = os.path.basename(path)[:-4]
                    if name[-4] == '-' and name[-2] == '0' and name[-1] == '2':
                        two = True
                if not two and os.path.exists(doc_path.replace('/tiku2', '/tiku4')):
                    doc_path = doc_path.replace('/tiku2', '/tiku4')
                    path = path.replace('/tiku2', '/tiku4')
                # else:
                #     doc_path = doc_path.replace('/tiku2', '/tiku')
                #     path = path.replace('/tiku2', '/tiku')
        if not os.path.exists(doc_path):
            print('not exists:', doc_path)
            progress_queue.put(('error', doc_path))
            continue
        if '/sjb/' in path:
            det_path = path.replace('/sjb/', '/jsons/')[:-4] + '.json'
        elif '/tiku4' in path or '/tiku5' in path or '/tiku6' in path:
            det_path = path[:-4] + '_det.json'
        else:
            det_path = os.path.join('/mnt/ceph2/datasets/tiku/jsons/images', os.path.basename(path)[:-4] + '.json')
        if not os.path.exists(det_path):
            print('not exists:', det_path)
            progress_queue.put(('error', det_path))
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
            img = cv2.imread(path)
            h, w = img.shape[:2]
            data = json.load(open(det_path, encoding='utf-8'))
            regions, pics, tables = get_regions(data, w, h)
            regions = sorted(regions, key=lambda x: (x['bbox'][3], x['bbox'][2]))
            ocr = ''
            for region in regions:
                ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
            if ocr.count('□') > 2 and '□' not in text2:
                progress_queue.put(('□', path))
                continue
            text3 = match_pics(text2, pics)
            if text3 is None:
                progress_queue.put(('unmatch', path))
                continue
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        input = f'OCR:\n{ocr}\n文档：'
        output = text3
        put_queue(progress_queue, input, output, tokenizer)

        suffixes = ['_crop_hw_1226', '_crop_hwn_1226',
                    '_crop_pp_1213', '_crop_ppn_1213',
                    '_crop_ppc_1225', '_crop_ppcn_1225']
        if any(s in path for s in suffixes):
            match = re.search(r'_crop_[a-z]+?_\d+?\.', path)
            suffix = match.group()[:-1]
            suffixes.remove(suffix)
            for s in suffixes:
                new_path = path.replace(suffix, s)
                det_path = new_path.replace('/sjb/', '/jsons/')[:-4] + '.json'
                if not os.path.exists(det_path):
                    print('not exists:', det_path)
                    progress_queue.put(('error', det_path))
                    continue
                data = json.load(open(det_path, encoding='utf-8'))
                regions, pics, tables = get_regions(data, w, h)
                regions = sorted(regions, key=lambda x: (x['bbox'][3], x['bbox'][2]))
                ocr = ''
                for region in regions:
                    ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
                text3 = match_pics(text2, pics)
                if text3 is None:
                    progress_queue.put(('unmatch', path))
                    continue
                input = f'OCR:\n{ocr}\n文档：'
                output = text3
                put_queue(progress_queue, input, output, tokenizer)


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
    files = glob('/mnt/ceph2/datasets/tiku2/words0/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku2/words0-2/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku2/words_split/**/*.docx', recursive=True)
    # files = ['/mnt/ceph2/datasets/tiku2/words0/小学/江苏/2019-2020学年苏教版一年级下册期末测试数学试卷4_973796_数学_1.docx']
    # print(len(files))
    # with open('exclude.txt', 'r', encoding='utf-8') as f:
    #     exclude = [line.strip() for line in f]
    # files = [f for f in files if os.path.basename(f) not in exclude]
    # print(len(files))
    files += glob('/mnt/ceph2/datasets/tiku6/words*/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku7/words*/*.docx', recursive=True)
    print(len(files))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, False, progress_queue, '/mnt/ceph2/LLaMA-Factory-main/datasets/conversations_tiku.json')
    else:
        process(files, tokenizer, True, False, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph2/datasets/tiku/sjb/0322/*.docx', recursive=True)
    files += glob(f'/mnt/ceph2/datasets/tiku/sjb/0401/*.docx', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det14.json')
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
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    prepare_tiku(tokenizer, parallel)
    # prepare_sjb(tokenizer, parallel)
    # prepare_camera(tokenizer, parallel)


def check():
    conversations = json.load(open('/mnt/ceph2/LLaMA-Factory-main/datasets/conversations_tiku.json', encoding='utf-8'))
    print(len(conversations))
    random.shuffle(conversations)
    for c in conversations[:10]:
        print(c)


if __name__ == "__main__":
    # prepare_imgs()
    # prepare()
    check()
