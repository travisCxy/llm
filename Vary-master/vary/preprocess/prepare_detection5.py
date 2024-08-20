import os
import json
import re
import random
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
    files0 = glob('/mnt/ceph2/datasets/tiku2/words0/**/*.docx', recursive=True)
    files0 += glob('/mnt/ceph2/datasets/tiku2/words0-2/**/*.docx', recursive=True)
    files0 += glob('/mnt/ceph2/datasets/tiku2/words_split/**/*.docx', recursive=True)
    files = []
    for src in tqdm(files0):
        if 'words0-2' in src:
            name = os.path.basename(src)[:-5]
            if name[-4] == '-' and name[-2] == '0' and name[-1] == '2':
                continue
        if os.path.exists(src.replace('/tiku2', '/tiku4')):
            src = src.replace('/tiku2', '/tiku4')
        else:
            src = src.replace('/tiku2', '/tiku')
        files.append(src)
    return files


def create_clos():
    files = get_docx()
    pbar = tqdm(total=len(files))
    success = 0
    for path in files:
        try:
            if 'tiku4' in path:
                dst = path.replace('/tiku4', '/tiku6')
            else:
                dst = path.replace('/tiku', '/tiku6')
            dir = os.path.dirname(dst)
            if not os.path.exists(dir):
                os.makedirs(dir)
            if create_clos_for_pic(path, dst):
                success += 1
            pbar.update(1)
            pbar.set_postfix(success=success)
        except Exception as e:
            print(path)
            continue
    print(success)


def prepare_imgs():
    files = glob('/mnt/ceph2/datasets/tiku5/words0/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku5/words0-2/**/*.docx', recursive=True)
    files += glob('/mnt/ceph2/datasets/tiku5/words_split/**/*.docx', recursive=True)
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
        if '/sjb/' not in path:
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
                else:
                    doc_path = doc_path.replace('/tiku2', '/tiku')
                    path = path.replace('/tiku2', '/tiku')
        if not os.path.exists(doc_path):
            print('not exists:', doc_path)
            progress_queue.put(('error', doc_path))
            continue
        if '/sjb/' in path:
            det_path = path.replace('/sjb/', '/jsons/')[:-4] + '.json'
        elif '/tiku4' in path or '/tiku5' in path:
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


def process_parallel(target, files, tokenizer, footer, with_font, progress_queue, output_path):
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=target, args=(chunk, tokenizer, footer, with_font, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'unmatch': 0, 'toomany': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'],
                                     unmatch=counts['unmatch'], toomany=counts['toomany'])
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
    # files = glob('/mnt/ceph2/datasets/tiku2/words0/**/*.docx', recursive=True)
    # files += glob('/mnt/ceph2/datasets/tiku2/words0-2/**/*.docx', recursive=True)
    # files += glob('/mnt/ceph2/datasets/tiku2/words_split/**/*.docx', recursive=True)
    files = ['/mnt/ceph2/datasets/tiku4/words_split/初中/浙江/2021年浙江省杭州市滨江区中考数学一模试卷_1112098_数学_3.docx']
    print(len(files))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det12.json')
    else:
        process(files, tokenizer, True, True, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph2/datasets/tiku/sjb/0322/*.docx', recursive=True)
    files += glob(f'/mnt/ceph2/datasets/tiku/sjb/0401/*.docx', recursive=True)
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(process, files, tokenizer, True, False, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_sjb_det12.json')
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
        process_parallel(process_camera, files, tokenizer, True, True, progress_queue, '/mnt/ceph2/datasets/tiku/vary/conversations_camera_det12.json')
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
    from convertor.docx2html import DocxConvertor
    docx = DocxConvertor()
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/vary/conversations_tiku_det12.json', encoding='utf-8'))
    random.seed(1)
    random.shuffle(conversations)
    print(len(conversations))
    for c in tqdm(conversations):
        path = '/mnt/ceph2/datasets/' + c['image']
        # if '2021-2022学年山东省济宁市微山县一年级（下）期末数学试卷_1341089_数学_2' not in path:
        #     continue
        if '数学' not in path:
            continue
        text = c['conversations'][1]['value']
        # if text.count('<footer') < 1:
        #     continue
        if '□' not in text:
            continue
        # if '□' not in text and '□' not in c['conversations'][0]['value']:
        #     continue
        # if '□' in text and '□' in c['conversations'][0]['value']:
        #     continue
        print(path)
        print(c['conversations'][0]['value'])
        with open(r'output/ocr.txt', 'w', encoding='utf-8') as f:
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
    # create_clos()
    # prepare_imgs()
    # prepare()
    check()

    # import shutil
    # files = glob('/mnt/ceph2/datasets/tiku2/words0/**/*.docx', recursive=True)
    # files += glob('/mnt/ceph2/datasets/tiku2/words0-2/**/*.docx', recursive=True)
    # files += glob('/mnt/ceph2/datasets/tiku2/words_split/**/*.docx', recursive=True)
    # for path in tqdm(files):
    #     dst = path.replace('words', 'images')[:-5] + '.jpg'
    #     src = dst.replace('/tiku2/', '/tiku/')
    #     dir = os.path.dirname(dst)
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #     shutil.copy(src, dst)
    #     src = path.replace('/tiku2/', '/tiku/')[:-5] + '.json'
    #     if os.path.exists(src):
    #         dst = dst[:-4] + '.json'
    #         shutil.copy(src, dst)
    # print(len(glob('/mnt/ceph2/datasets/tiku2/words*/**/*.docx', recursive=True)))
    # print(len(glob('/mnt/ceph2/datasets/tiku4/words*/**/*.docx', recursive=True)))
    # print(len(glob('/mnt/ceph2/datasets/tiku5/words*/**/*.docx', recursive=True)))
    # docx2img(['/mnt/ceph2/datasets/tiku4/2016-2017学年湖北省武汉市钢城十一中九年级（上）月考数学试卷（9月份）_21323_数学_1-200.docx'])

    # path = '/mnt/ceph2/datasets/tiku4/2016-2017学年湖北省武汉市钢城十一中九年级（上）月考数学试卷（9月份）_21323_数学_1-200.docx'
    # from spire.doc import Document, FileFormat
    # document = Document()
    # document.LoadFromFile(path)
    # text = document.GetText()
    # count = document.GetPageCount()
    # print(count)
    # document.SaveToFile(path[:-5]+'-2.pdf', FileFormat.PDF)
    # document.Close()
    #
    # import fitz
    # from PIL import Image
    #
    # doc = fitz.open(str(path[:-5] + '-2.pdf'))
    # page = doc.load_page(0)
    # pix = page.get_pixmap(dpi=random.randint(150, 300))
    # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    # img.save(path[:-5] + '-2.jpg')
