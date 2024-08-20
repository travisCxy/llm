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
from convertor.common import get_regions, match_pics, sort_regions
from glob import glob
from tqdm import tqdm


def get_docx():
    files = glob('/mnt/ceph2/datasets/tiku/words1/**/*.docx', recursive=True)
    docx = DocxConvertor(with_font=False)
    count = 0
    with open('/mnt/ceph2/datasets/tiku6/files.txt', 'w', encoding='utf-8') as f:
        for path in tqdm(files):
            if '数学' not in path:
                continue
            try:
                text = docx.docx2html(path, format=2)
            except Exception as e:
                continue
            if text.count('\sqrt') > 3:
                f.write(path + '\n')
                count += 1
                if count >= 5000:
                    break
    print(count)


def prepare_imgs():
    files = glob('/mnt/ceph2/datasets/tiku6/words*/**/*.docx', recursive=True)
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
            regions.extend(pics)
            regions.extend(tables)
            regions = sort_regions(regions)
            ocr = ''
            for region in regions:
                ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
            if ocr.count('□') > 2 and '□' not in text2:
                progress_queue.put(('□', path))
                continue
            text3 = match_pics(text2, pics, box=True)
            if text3 is None:
                progress_queue.put(('unmatch', path))
                continue
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        if with_font:
            text1 = f'<image>\nConvert with font:\n'
        else:
            text1 = f'<image>\nConvert:\n'
        put_queue(progress_queue, path, text1, text3, tokenizer)
        if with_font:
            text1 = f'<image>\nOCR and Convert with font:\n'
            text3 = 'OCR:\n' + ocr + '\nConvert with font:\n' + text3
        else:
            text1 = f'<image>\nOCR and Convert:\n'
            text3 = 'OCR:\n' + ocr + '\nConvert:\n' + text3
        put_queue(progress_queue, path, text1, text3, tokenizer)

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
                regions.extend(pics)
                regions.extend(tables)
                regions = sort_regions(regions)
                ocr = ''
                for region in regions:
                    ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
                text3 = match_pics(text2, pics, box=True)
                if text3 is None:
                    progress_queue.put(('unmatch', path))
                    continue
                if with_font:
                    text1 = f'<image>\nConvert with font:\n'
                else:
                    text1 = f'<image>\nConvert:\n'
                put_queue(progress_queue, path, text1, text3, tokenizer)
                if with_font:
                    text1 = f'<image>\nOCR and Convert with font:\n'
                    text3 = 'OCR:\n' + ocr + '\nConvert with font:\n' + text3
                else:
                    text1 = f'<image>\nOCR and Convert:\n'
                    text3 = 'OCR:\n' + ocr + '\nConvert:\n' + text3
                put_queue(progress_queue, new_path, text1, text3, tokenizer)


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
    progress_bar = tqdm(total=len(files)*2)
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
    print(len(files))
    with open('exclude.txt', 'r', encoding='utf-8') as f:
        exclude = [line.strip() for line in f]
    files = [f for f in files if os.path.basename(f) not in exclude]
    print(len(files))
    files += glob('/mnt/ceph2/datasets/tiku6/words*/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku7/words*/*.docx', recursive=True)
    print(len(files))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det_ocr2.json')
    else:
        process(files, tokenizer, True, True, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph2/datasets/tiku/sjb/0322/*.docx', recursive=True)
    files += glob(f'/mnt/ceph2/datasets/tiku/sjb/0401/*.docx', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det_ocr2.json')
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
            regions.extend(pics)
            regions.extend(tables)
            regions = sort_regions(regions)
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
                text2 = text2[:i] + f'<box>({pic["bbox"][0]},{pic["bbox"][1]}),({pic["bbox"][2]},{pic["bbox"][3]})</box>' + text2[i + 5:]
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        if with_font:
            text1 = f'<image>\nConvert with font:\n'
        else:
            text1 = f'<image>\nConvert:\n'
        put_queue(progress_queue, path, text1, text2, tokenizer)
        if with_font:
            text1 = f'<image>\nOCR and Convert with font:\n'
            text2 = 'OCR:\n' + ocr + '\nConvert with font:\n' + text2
        else:
            text1 = f'<image>\nOCR and Convert:\n'
            text2 = 'OCR:\n' + ocr + '\nConvert:\n' + text2
        put_queue(progress_queue, path, text1, text2, tokenizer)


def prepare_camera(tokenizer, parallel):
    files = glob('/mnt/ceph2/datasets/tiku5/words_c/**/*.png', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku5/words_c/**/*.jpg', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process_camera, files, tokenizer, True, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_camera_det_ocr2.json')
    else:
        process_camera(files, tokenizer, True, True, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    # prepare_tiku(tokenizer, parallel)
    prepare_sjb(tokenizer, parallel)
    # prepare_camera(tokenizer, parallel)


def check():
    from convertor.docx2html import DocxConvertor
    docx = DocxConvertor()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det_ocr2.json', encoding='utf-8'))
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
        # path = '/mnt/ceph2/datasets/YYT_DET_20210602/' + c['image']
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
        # if '<pic' not in text2:
        #     continue
        if 'OCR' in text1:
            continue
        print(path)
        print(text1)
        print(text2)
        print(len(text2))
        # with open(r'output/ocr.txt', 'w', encoding='utf-8') as f:
        #     f.write(c['conversations'][0]['value'])
        if '<head>' in text2:
            text2 = text2[text2.find('<head>'):]
        else:
            text2 = text2[text2.find('<body>'):]
        pretty_html = docx.pretty(text2, 2, path)
        with open(r'test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', text2)
        for match in matches:
            x0, y0, x1, y1 = map(int, match)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        img = cv2.resize(img, (768, 768))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    # prepare()
    check()

    # conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_yyt_ocr.json', encoding='utf-8'))
    # for c in tqdm(conversations):
    #     # c['conversations'][0]['value'] = '<image>\nOCR:\n'
    #     c['conversations'][1]['value'] = c['conversations'][1]['value'].replace('</box>', '</box>\n')
    #     # text1 = c['conversations'][0]['value']
    #     # text2 = c['conversations'][1]['value']
    #     # print(text1)
    #     # print(text2)
    # with open('/mnt/ceph2/datasets/tiku/vary/conversations_yyt_ocr2.json', 'w', encoding='utf-8') as f:
    #     json.dump(conversations, f, ensure_ascii=False, indent=2)
