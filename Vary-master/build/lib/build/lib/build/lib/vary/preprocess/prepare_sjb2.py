import os
import json
import re
import random
import shutil
import transformers
import multiprocessing
from docx import Docx
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def prepare_html():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    files = glob(f'/mnt/ceph/tiku/sjb/0322/*.docx', recursive=True)
    conversations = []
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'toosmall': 0, 'nobox': 0}
    with open(f'/mnt/ceph/tiku/sjb/error.txt', 'w', encoding='utf-8') as f:
        docx = Docx()
        template = json.load(open(f'template_html.json', encoding='utf-8'))
        for path in tqdm(files):
            if '200896015-1' not in path:
                continue
            name = path.replace(f'/mnt/ceph/tiku/sjb/', '')[:-5]+'.jpg'
            try:
                text2 = docx.docx2html(path, format=2)
                matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
                skip = False
                boxes = {}
                for match in matches:
                    if float(match[3]) < 0.1:
                        skip = True
                        break
                    # centery = (float(match[2]) + float(match[4])) / 2
                    text = f'@图{len(boxes)}@'
                    boxes[text] = [float(x) for x in match[1:]]
                    text2 = text2.replace(match[0], text)
                if skip:
                    counts['toosmall'] += 1
                    continue
            except Exception as e:
                print(e)
                f.write(path + '\n')
                counts['error'] += 1
                continue
            match = re.search(r'_crop_[a-z]+?_\d+?\.', name)
            name = name[:match.start()]
            for suffix in ['_crop_hw_1226', '_crop_hwn_1226',
                           '_crop_pp_1213', '_crop_ppn_1213',
                           '_crop_ppc_1225', '_crop_ppcn_1225']:
                if len(boxes) > 0 and (suffix == '_crop_hw_1226' or 'n' in suffix):
                    continue
                text1 = '<image>\n' + random.choice(template)
                text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
                tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
                if len(tokenized.input_ids[0]) > 2048:
                    counts['toolong'] += 1
                    continue
                src = f'/mnt/ceph/tiku/sjb/{name + suffix + ".jpg"}'
                assert os.path.exists(src)
                dst = src.replace('sjb', 'sjb2')
                if not os.path.exists(os.path.dirname(dst)):
                    os.makedirs(os.path.dirname(dst))
                if len(boxes) > 0:
                    image = Image.open(src).convert('RGB')
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.truetype("SIMSUN.TTC", size=48)
                    width, height = image.size
                    skip = False
                    for text, box in boxes.items():
                        x = int(box[0] * width)
                        y = int(box[1] * height)
                        w = int(box[2] * width)
                        h = int(box[3] * height)
                        draw.rectangle([x, y, x + w, y + h], outline='red', fill='white', width=2)
                        center = (x + w // 2, y + h // 2)
                        text_width, text_height = font.getbbox(text)[2:]
                        if text_width >= w or text_height >= h:
                            font = ImageFont.truetype("SIMSUN.TTC", size=24)
                            text_width, text_height = font.getbbox(text)[2:]
                            if text_width >= w or text_height >= h:
                                skip = True
                                break
                        draw.text((center[0] - text_width // 2, center[1] - text_height // 2), text, font=font, fill='red')
                    if skip:
                        print('too small')
                        continue
                    image.save(dst)
                else:
                    shutil.copy(src, dst)
                conversations.append({"image": name + suffix + '.jpg',
                                      "conversations":
                                          [{"from": "human", "value": text1},
                                           {"from": "gpt", "value": text2}]})
                counts['success'] += 1
    print(len(conversations), counts)
    with open(f'/mnt/ceph/tiku/sjb2/conversations_html.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def process(files, progress_queue):
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    docx = Docx()
    template = json.load(open(f'template_html.json', encoding='utf-8'))
    for path in tqdm(files):
        src = path
        name = path.replace(f'/mnt/ceph/tiku/', '')
        path = path.replace('/images', '/words')[:-4] + '.docx'
        assert os.path.exists(path)
        try:
            text2 = docx.docx2html(path, format=2)
            matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
            skip = False
            boxes = {}
            for match in matches:
                if float(match[3]) < 0.1:
                    skip = True
                    break
                text = f'@图{len(boxes)}@'
                boxes[text] = [float(x) for x in match[1:]]
                text2 = text2.replace(match[0], text)
            if skip:
                progress_queue.put(('toosmall', path))
                continue
        except Exception as e:
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        text1 = '<image>\n' + random.choice(template)
        text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
        tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
        if len(tokenized.input_ids[0]) > 2048:
            progress_queue.put(('toolong', path))
            continue
        dst = src.replace('/tiku/', '/tiku/sjb2/')
        if len(boxes) > 0:
            image = Image.open(src).convert('RGB')
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("SIMSUN.TTC", size=48)
            width, height = image.size
            skip = False
            for text, box in boxes.items():
                x = int(box[0] * width)
                y = int(box[1] * height)
                w = int(box[2] * width)
                h = int(box[3] * height)
                draw.rectangle([x, y, x + w, y + h], outline='red', fill='white', width=2)
                center = (x + w // 2, y + h // 2)
                text_width, text_height = font.getbbox(text)[2:]
                if text_width >= w or text_height >= h:
                    font = ImageFont.truetype("SIMSUN.TTC", size=24)
                    text_width, text_height = font.getbbox(text)[2:]
                    if text_width >= w or text_height >= h:
                        skip = True
                        break
                draw.text((center[0] - text_width // 2, center[1] - text_height // 2), text, font=font, fill='red')
            if skip:
                print('too small')
                continue
            image.save(dst)
        else:
            shutil.copy(src, dst)
        progress_queue.put(('success', {"image": name,
                                        "conversations":
                                            [{"from": "human", "value": text1},
                                             {"from": "gpt", "value": text2}]}))


def prepare_tiku_html():
    files = glob(f'/mnt/ceph/tiku/images0/**/*.jpg', recursive=True)
    random.shuffle(files)
    files0 = files[:2000]

    files = glob(f'/mnt/ceph/tiku/images0-2/**/*.jpg', recursive=True)
    random.shuffle(files)
    files02 = []
    for path in files:
        name = os.path.basename(path)[:-4]
        if name[-4] != '-':
            continue
        if name[-2] == '0' and name[-1] == '0':
            files02.append(path)
            if len(files02) >= 2000:
                break
    for path in files:
        name = os.path.basename(path)[:-4]
        if name[-4] != '-':
            continue
        if name[-2] == '0' and name[-1] == '2':
            files02.append(path)
            if len(files02) >= 4000:
                break

    with open('/mnt/ceph/tiku/images_split/files.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    files = [f'/mnt/ceph/tiku/images_split/{file}' for file in files if file]
    random.shuffle(files)
    files = files[:2000]

    files = files0 + files02 + files

    for path in files:
        path = path.replace('/tiku/', '/tiku/sjb2/')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    # progress_queue = multiprocessing.Queue()
    # process(files, progress_queue)
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    progress_queue = multiprocessing.Queue()
    runners = [multiprocessing.Process(target=process, args=(chunk, progress_queue)) for chunk in file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'toosmall': 0}
    conversations = []
    while not progress_bar.n >= len(files):
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], toomanny=counts['toolong'], toosmall=counts['toosmall'])
        except Exception as e:
            if all(not runner.is_alive() for runner in runners):
                break
            continue
        progress_bar.update(1)
    progress_bar.close()
    print(len(conversations), counts)
    with open(f'/mnt/ceph/tiku/sjb2/conversations_tiku_html.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def check():
    import re
    import cv2
    docx = Docx()
    conversations = json.load(open('/mnt/ceph/tiku/sjb2/conversations_tiku_html.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        # if '200680218-1' not in c['image']:
        #     continue
        path = os.path.join('/mnt/ceph/tiku/sjb2', c['image'])
        # if not os.path.exists(path):
        #     continue
        text = c['conversations'][1]['value']
        print(path)
        print(c['conversations'][0]['value'])
        print(text)
        pretty_html = docx.pretty(text, 2, path)
        with open(r'/mnt/ceph/Vary/images/test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (768, 768))
        matches = re.findall(r'@\[.*?\]@', c['conversations'][1]['value'])
        boxes = []
        for match in matches:
            match = match[2:-2]
            boxes.append([float(x) / 1024 for x in match.split(', ')])
        for box in boxes:
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0 * 768), int(y0 * 768), int(x1 * 768), int(y1 * 768)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()

