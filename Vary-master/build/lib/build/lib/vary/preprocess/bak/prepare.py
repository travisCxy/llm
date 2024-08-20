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


def draw_boxes(path, boxes, titles):
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    skip = False
    for title, box in zip(titles, boxes):
        box = box[1]
        x = int(box[0] * width)
        y = int(box[1] * height)
        w = int(box[2] * width)
        h = int(box[3] * height)
        draw.rectangle([x, y, x + w, y + h], outline='green', fill='green', width=1)
        center = (x + w // 2, y + h // 2)
        size = random.randint(36, 96)
        while size >= 36:
            font = ImageFont.truetype("SIMSUN.TTC", size=size)
            text_width, text_height = font.getbbox(title)[2:]
            if text_width < w and text_height < h:
                break
            size = size // 2
        if size < 36:
            skip = True
            break
        draw.text((center[0] - text_width // 2, center[1] - text_height // 2), title, font=font, fill='red')
    if skip:
        return None
    return image


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
    template = json.load(open(f'template3.json', encoding='utf-8'))
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
            boxes = []
            for match in matches:
                if float(match[3]) < 0.05:
                    skip = True
                    break
                # x1 = int(float(match[1]) * 1000)
                # y1 = int(float(match[2]) * 1000)
                # x2 = int((float(match[1]) + float(match[3])) * 1000)
                # y2 = int((float(match[2]) + float(match[4])) * 1000)
                # text2 = text2.replace(match[0], f'<box>({x1},{y1}),({x2},{y2})</box>')
                # text = f'图{len(boxes)}'
                # boxes[text] = [float(x) for x in match[1:]]
                # text2 = text2.replace(match[0], f'<box>{text}</box>')
                boxes.append((match[0], [float(x) for x in match[1:]]))
            if skip:
                progress_queue.put(('toosmall', path))
                continue
        except Exception as e:
            print(path)
            print(e)
            progress_queue.put(('error', path + '\n' + str(e)))
            continue
        text1 = '<image>\n' + random.choice(template)
        random.shuffle(boxes)
        text3 = text2
        titles = []
        for i, box in zip(random.sample(range(1, 99), len(boxes)), boxes):
            titles.append(f'{i:02d}')
            text3 = text3.replace(box[0], f'<box>{titles[-1]}</box>')
        source = {"image": path.replace('/mnt/ceph/tiku/', ''),
                  "conversations":
                      [{"from": "human", "value": text1},
                       {"from": "gpt", "value": text3}]}
        input_id = preprocess(source, tokenizer)
        if len(input_id) > 2048:
            progress_queue.put(('toolong', path))
            continue
        dst = path.replace('/tiku/', '/tiku/mix/')
        if len(boxes) > 0:
            image = draw_boxes(path, boxes, titles)
            if image is None:
                progress_queue.put(('toosmall', path))
                continue
            image.save(dst)
        else:
            shutil.copy(path, dst)
        progress_queue.put(('success', source))

        suffixes = ['_crop_hw_1226', '_crop_hwn_1226',
                    '_crop_pp_1213', '_crop_ppn_1213',
                    '_crop_ppc_1225', '_crop_ppcn_1225']
        if any(s in path for s in suffixes):
            match = re.search(r'_crop_[a-z]+?_\d+?\.', path)
            suffix = match.group()[:-1]
            if len(boxes) > 0 and '_hw_' in suffix:
                continue
            suffixes.remove(suffix)
            for s in suffixes:
                if len(boxes) > 0 and (s == '_crop_hw_1226' or 'n' in s):
                    continue
                new_path = path.replace(suffix, s)
                text1 = '<image>\n' + random.choice(template)
                random.shuffle(boxes)
                text3 = text2
                titles = []
                for i, box in zip(random.sample(range(1, 99), len(boxes)), boxes):
                    titles.append(f'{i:02d}')
                    text3 = text3.replace(box[0], f'<box>{titles[-1]}</box>')
                source = {"image": new_path.replace('/mnt/ceph/tiku/', ''),
                          "conversations":
                              [{"from": "human", "value": text1},
                               {"from": "gpt", "value": text3}]}
                dst = new_path.replace('/tiku/', '/tiku/mix/')
                if len(boxes) > 0:
                    image = draw_boxes(path, boxes, titles)
                    if image is None:
                        progress_queue.put(('toosmall', path))
                        continue
                    image.save(dst)
                else:
                    shutil.copy(new_path, dst)
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
    with open(f'/mnt/ceph/tiku/mix/small.txt', 'w', encoding='utf-8') as f:
        while True:
            try:
                result = progress_queue.get(timeout=5)
                counts[result[0]] += 1
                if result[0] == 'success':
                    conversations.append(result[1])
                progress_bar.set_postfix(count=counts['success'], error=counts['error'], toolong=counts['toolong'],
                                         toosmall=counts['toosmall'])
                if result[0] == 'toosmall':
                    f.write(result[1] + '\n')
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
    files = glob(f'/mnt/ceph/tiku/images0/**/*.jpg', recursive=True)
    random.shuffle(files)
    files0 = files[:10000]

    files = glob(f'/mnt/ceph/tiku/images0-2/**/*.jpg', recursive=True)
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

    with open('/mnt/ceph/tiku/images_split/files.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    files = [f'/mnt/ceph/tiku/images_split/{file}' for file in files if file]
    random.shuffle(files)
    files = files[:30000]

    files = files0 + files02 + files
    for path in files:
        path = path.replace('/tiku/', '/tiku/mix/')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, False, progress_queue, '/mnt/ceph/tiku/mix/conversations_tiku.json')
    else:
        process(files, tokenizer, False, progress_queue)


def prepare_sjb(tokenizer, parallel):
    files = glob(f'/mnt/ceph/tiku/sjb/0322/*.docx', recursive=True)
    for path in files:
        path = path.replace('/tiku/', '/tiku/mix/')
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, tokenizer, True, progress_queue, '/mnt/ceph/tiku/mix/conversations_sjb.json')
    else:
        process(files, tokenizer, True, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    prepare_tiku(tokenizer, parallel)
    # prepare_sjb(tokenizer, parallel)


def merge():
    conversations = []
    for path in glob('/mnt/ceph/tiku/qwen/conversations_*.json'):
        conversations.extend(json.load(open(path, encoding='utf-8')))
    print(len(conversations))
    random.shuffle(conversations)
    with open('/mnt/ceph/tiku/qwen/conversations.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def check():
    import re
    import cv2
    docx = Docx()
    conversations = json.load(open('/mnt/ceph/tiku/mix/conversations_tiku.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        text = c['conversations'][1]['value']
        if '<box' not in text:
            continue
        path = '/mnt/ceph/tiku/mix/' + c['image']
        if '2010-2011学年上海市新场中学八年级（下）第一次月考数学试卷_20803_数学_3.jpg' not in path:
            continue
        print(path)
        print(c['conversations'][0]['value'])
        print(text)
        pretty_html = docx.pretty(text, 2, path)
        with open(r'/mnt/ceph/tiku/qwen/test.html', 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        img = cv2.imread(path)
        img = cv2.resize(img, (768, 768))
        # matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', c['conversations'][1]['value'])
        # for match in matches:
        #     x0, y0, x1, y1 = map(int, match)
        #     cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()
    # import cv2
    # with open('/mnt/ceph/tiku/small.txt', 'r', encoding='utf-8') as f:
    #     paths = f.read().split('\n')
    # for path in paths:
    #     path = path.replace('/words', '/images')[:-5] + '.jpg'
    #     print(path)
    #     img = cv2.imread(path)
    #     img = cv2.resize(img, (768, 768))
    #     cv2.imshow('img', img)
    #     if cv2.waitKey(0) == 27:
    #         break
