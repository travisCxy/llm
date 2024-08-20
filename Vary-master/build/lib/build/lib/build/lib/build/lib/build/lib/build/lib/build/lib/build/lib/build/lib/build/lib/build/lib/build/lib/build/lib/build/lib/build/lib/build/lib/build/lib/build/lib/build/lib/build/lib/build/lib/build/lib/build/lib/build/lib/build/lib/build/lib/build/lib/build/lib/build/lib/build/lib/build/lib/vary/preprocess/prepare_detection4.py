import os
import json
import re
import random
import sys
import cv2
import transformers
import multiprocessing
from convertor.docx2html import DocxConvertor
from glob import glob
from tqdm import tqdm


def preapre_docx():
    import shutil
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        path = '/mnt/ceph2/datasets/tiku/' + c['image']
        src = path.replace('/images', '/words')[:-4] + '.docx'
        dst = src.replace('/tiku', '/tiku2')
        dir = os.path.dirname(dst)
        if not os.path.exists(dir):
            os.makedirs(dir)
        shutil.copy(src, dst)


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
        if '/sjb/' not in path:
            two = False
            if 'words0-2' in doc_path:
                name = os.path.basename(path)[:-4]
                if name[-4] == '-' and name[-2] == '0' and name[-1] == '2':
                    two = True
            if not two and os.path.exists(doc_path.replace('/tiku2', '/tiku4')):
                doc_path = doc_path.replace('/tiku2', '/tiku4')
                path = path.replace('/tiku2', '/tiku4')
            else:
                doc_path = doc_path.replace('/tiku2', '/tiku')
                path = path.replace('/tiku2', '/tiku')
        if not os.path.exists(doc_path):
            print('not exists:', doc_path)
            progress_queue.put(('error', doc_path))
            continue
        if '/sjb/' in path:
            det_path = path.replace('/sjb/', '/jsons/')[:-4] + '.json'
        elif '/tiku4' in path:
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
    files = glob('/mnt/ceph2/datasets/tiku2/words0/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku2/words0-2/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku2/words_split/**/*.docx', recursive=True)
    print(len(files))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, False, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det9.json')
    else:
        process(files, tokenizer, False, True, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph2/datasets/tiku/sjb/0322/*.docx', recursive=True)
    files += glob(f'/mnt/ceph2/datasets/tiku/sjb/0401/*.docx', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, True, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det9.json')
    else:
        process(files, tokenizer, True, False, progress_queue)


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
    import zipfile
    docx = Docx()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det9.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        # if 'tiku/' not in c['image']:
        #     continue
        path = '/mnt/ceph2/datasets/' + c['image']
        # if '200895764-1_crop_hw_1226' not in path:
        #     continue
        # if 'split' not in path:
        #     continue
        # skip = True
        # docx_path = path.replace('/images', '/words')[:-4] + '.docx'
        # with zipfile.ZipFile(docx_path, 'r') as z:
        #     xml_content = z.read('word/document.xml').replace(b'\xe2\x80\x8b', b'').decode('utf-8')
        #     if '<w:tbl ' in xml_content and '<w:tblGrid>' not in xml_content:
        #         skip = False
        # if skip:
        #     continue
        text = c['conversations'][1]['value']
        if 'ruby' not in text:
            continue
        # matches = re.findall(r'（(.*?)）', text)
        # skip = True
        # for match in matches:
        #     if re.match(r'^_+$', match):
        #         skip = False
        #         break
        # if skip:
        #     continue
        # if 'table' not in text:
        #     continue
        # if 'indent:0.0' not in text:
        #     continue
        print(path)
        # if 'OCR' in c['conversations'][0]['value']:
        #     continue
        print(c['conversations'][0]['value'])
        with open(r'output/ocr.txt', 'w', encoding='utf-8') as f:
            f.write(c['conversations'][0]['value'])
        print(text)
        pretty_html = docx.pretty(text, 2, path)
        with open(r'output/test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (768, 768))
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    # check()

    from convertor.common import get_regions
    data = json.load(open('output/tmp.json', encoding='utf-8'))
    img = cv2.imread('output/tmp.jpg')
    regions, pics, tables = get_regions(data, 1000, 1000, merge=True)
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
    for pic in pics:
        bbox = pic['bbox']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(img, pic['result'], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    img = cv2.resize(img, (768, 768))
    cv2.imshow('img', img)
    cv2.waitKey(0)
