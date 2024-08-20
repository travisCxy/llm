import os
import json
import re
import random
import sys
import cv2
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


def get_bbox(region_3point, w, h):
    xs = [region_3point[0], region_3point[2], region_3point[4]]
    ys = [region_3point[1], region_3point[3], region_3point[5]]
    x0 = int(min(xs) * 1000 / w)
    x1 = int(max(xs) * 1000 / w)
    y0 = int(min(ys) * 1000 / h)
    y1 = int(max(ys) * 1000 / h)
    return [x0, y0, x1, y1]


def rectangle_overlap_percentage(rect1, rect2):
    """
    计算第一个矩形在第二个矩形内的百分比。
    矩形格式：[x0, y0, x1, y1]，其中 (x0, y0) 是左上角坐标，(x1, y1) 是右下角坐标。
    """
    # 交集左上角坐标
    ix0 = max(rect1[0], rect2[0])
    iy0 = max(rect1[1], rect2[1])

    # 交集右下角坐标
    ix1 = min(rect1[2], rect2[2])
    iy1 = min(rect1[3], rect2[3])

    # 交集面积
    if ix1 >= ix0 and iy1 >= iy0:
        intersection_area = (ix1 - ix0) * (iy1 - iy0)
    else:
        intersection_area = 0  # 没有交集

    # 第一个矩形的面积
    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])

    # 计算第一个矩形在第二个矩形内的百分比
    if rect1_area > 0:
        percentage = (intersection_area / rect1_area) * 100
    else:
        percentage = 0  # 如果第一个矩形没有面积

    return percentage


def get_regions(data, w, h):
    regions = []
    for item in data['regions']:
        if item['cls'] not in [1, 10]:
            continue
        if item['result'][0] == '':
            continue
        bbox = get_bbox(item['region_3point'], w, h)
        regions.append({'bbox': bbox, 'result': item['result'][0]})
    pics = []
    for i, pic in enumerate(data['pics']):
        bbox = get_bbox(pic['region_3point'], w, h)
        pics.append({'bbox': bbox, 'result': f'<pic id="{i}"/>'})
        regions = [region for region in regions if rectangle_overlap_percentage(region['bbox'], bbox) < 80]
    tables = []
    for table in data['tables']:
        rect = table['rect']
        bbox = [int(rect[0] * 1000 / w), int(rect[1] * 1000 / h), int(rect[2] * 1000 / w), int(rect[3] * 1000 / h)]
        cells = ''
        for cell in table['cells']:
            texts = ''
            for text in cell['texts']:
                if 'content' not in text:
                    continue
                texts += f"<text>{text['content']}</text>"
            if texts:
                cells += f'<cell>{texts}</cell>'
        tables.append({'bbox': bbox, 'result': f'<table>{cells}</table>'})
        regions = [region for region in regions if rectangle_overlap_percentage(region['bbox'], bbox) < 80]
        pics = [pic for pic in pics if rectangle_overlap_percentage(pic['bbox'], bbox) < 80]
    regions.extend(pics)
    regions.extend(tables)
    regions = sorted(regions, key=lambda x: x['bbox'][3])
    return regions, pics, tables


def match_pics(text2, pics):
    matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
    used = [False] * len(pics)
    unmatch = False
    for match in matches:
        x0 = int(float(match[1]) * 1000)
        y0 = int(float(match[2]) * 1000)
        x1 = int((float(match[1]) + float(match[3])) * 1000)
        y1 = int((float(match[2]) + float(match[4])) * 1000)
        text = ''
        for i, pic in enumerate(pics):
            percentage = rectangle_overlap_percentage(pic['bbox'], [x0, y0, x1, y1])
            if percentage < 80:
                continue
            if used[i]:
                unmatch = True
                break
            text += pic['result']
            used[i] = True
        if text == '':
            unmatch = True
        if unmatch:
            break
        text2 = text2.replace(match[0], text)
    if unmatch:
        return None
    return text2


def put_queue(queue, path, text1, text2, tokenizer):
    source = {"image": path.replace('/mnt/ceph2/datasets/tiku/', ''),
              "conversations":
                  [{"from": "human", "value": text1},
                   {"from": "gpt", "value": text2}]}
    input_id = preprocess(source, tokenizer)
    if len(input_id) > 3600:
        queue.put(('toolong', path))
    else:
        queue.put(('success', source))


def process(files, tokenizer, footer, with_font, progress_queue):
    docx = Docx(with_font=with_font)
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
        if '/sjb/' in path:
            det_path = path.replace('/sjb/', '/jsons/')[:-4] + '.json'
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
            img = cv2.imread(path)
            h, w = img.shape[:2]
            data = json.load(open(det_path, encoding='utf-8'))
            regions, pics, tables = get_regions(data, w, h)
            ocr = ''
            for region in regions:
                ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
            text2 = docx.docx2html(doc_path, format=2, box_path=box_path, footer=footer)
            text3 = match_pics(text2, pics)
            if text3 is None:
                progress_queue.put(('unmatch', path))
                continue
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        if with_font:
            text1 = f'<image>\nOCR:\n{ocr}\nConvert with font:'
        else:
            text1 = f'<image>\nOCR:\n{ocr}\nConvert:'
        put_queue(progress_queue, path, text1, text3, tokenizer)
        if with_font:
            text1 = f'<image>\nConvert with font:'
        else:
            text1 = f'<image>\nConvert:'
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
                ocr = ''
                for region in regions:
                    ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
                text3 = match_pics(text2, pics)
                if text3 is None:
                    progress_queue.put(('unmatch', path))
                    continue
                if with_font:
                    text1 = f'<image>\nOCR:\n{ocr}\nConvert with font:'
                else:
                    text1 = f'<image>\nOCR:\n{ocr}\nConvert:'
                put_queue(progress_queue, new_path, text1, text3, tokenizer)
                if with_font:
                    text1 = f'<image>\nConvert with font:'
                else:
                    text1 = f'<image>\nConvert:'
                put_queue(progress_queue, new_path, text1, text3, tokenizer)


def process_parallel(files, tokenizer, footer, with_font, progress_queue, output_path):
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, tokenizer, footer, with_font, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'unmatch': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'],
                                     unmatch=counts['unmatch'])
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
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku.json', encoding='utf-8'))
    print(len(conversations))
    files = []
    for c in conversations:
        path = '/mnt/ceph2/datasets/tiku/' + c['image']
        files.append(path)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, False, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det4.json')
    else:
        process(files, tokenizer, False, True, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph2/datasets/tiku/sjb/0322/*.docx', recursive=True)
    files += glob(f'/mnt/ceph2/datasets/tiku/sjb/0401/*.docx', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, True, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det4.json')
    else:
        process(files, tokenizer, True, False, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    # prepare_tiku(tokenizer, parallel)
    prepare_sjb(tokenizer, parallel)


def show():
    import cv2
    for path in tqdm(glob('/mnt/ceph2/datasets/tiku/jsons/0401/*')):
        data = json.load(open(path, encoding='utf-8'))
        path = path.replace('jsons', 'sjb')[:-5] + '.jpg'
        print(path)
        img = cv2.imread(path)
        regions, pics, tables = get_regions(data, 1000, 1000)
        for region in tables:
            print(region['result'])
            x0, y0, x1, y1 = region['bbox']
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        # for region in data['regions']:
        #     if region['cls'] not in [1, 10]:
        #         continue
        #     region_3point = region['region_3point']
        #     xs = [region_3point[0], region_3point[2], region_3point[4]]
        #     ys = [region_3point[1], region_3point[3], region_3point[5]]
        #     x0 = int(min(xs))
        #     x1 = int(max(xs))
        #     y0 = int(min(ys))
        #     y1 = int(max(ys))
        #     cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        # for pic in data['pics']:
        #     region_3point = pic['region_3point']
        #     xs = [region_3point[0], region_3point[2], region_3point[4]]
        #     ys = [region_3point[1], region_3point[3], region_3point[5]]
        #     x0 = int(min(xs))
        #     x1 = int(max(xs))
        #     y0 = int(min(ys))
        #     y1 = int(max(ys))
        #     cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        # for table in data['tables']:
        #     rect = table['rect']
        #     x0, y0, x1, y1 = rect
        #     cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
        #     for cell in table['cells']:
        #         print(cell)
        img = cv2.resize(img, (768, 768))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


def check():
    import zipfile
    docx = Docx()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det3.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        path = '/mnt/ceph2/datasets/tiku/' + c['image']
        if 'split' not in path:
            continue
        # skip = True
        # docx_path = path.replace('/images', '/words')[:-4] + '.docx'
        # with zipfile.ZipFile(docx_path, 'r') as z:
        #     xml_content = z.read('word/document.xml').replace(b'\xe2\x80\x8b', b'').decode('utf-8')
        #     if '<w:tbl ' in xml_content and '<w:tblGrid>' not in xml_content:
        #         skip = False
        # if skip:
        #     continue
        text = c['conversations'][1]['value']
        # matches = re.findall(r'（(.*?)）', text)
        # skip = True
        # for match in matches:
        #     if re.match(r'^_+$', match):
        #         skip = False
        #         break
        # if skip:
        #     continue
        if 'table' not in text:
            continue
        # if 'indent:0.0' not in text:
        #     continue
        print(path)
        # if 'OCR' in c['conversations'][0]['value']:
        #     continue
        print(c['conversations'][0]['value'])
        with open(r'ocr.txt', 'w', encoding='utf-8') as f:
            f.write(c['conversations'][0]['value'])
        print(text)
        pretty_html = docx.pretty(text, 2, path)
        with open(r'test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (768, 768))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()
