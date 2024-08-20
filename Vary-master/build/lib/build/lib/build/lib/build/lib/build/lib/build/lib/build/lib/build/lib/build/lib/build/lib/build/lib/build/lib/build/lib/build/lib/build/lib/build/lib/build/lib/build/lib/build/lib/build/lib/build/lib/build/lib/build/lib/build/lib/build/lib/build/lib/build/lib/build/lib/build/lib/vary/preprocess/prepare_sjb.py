import os
import json
import re
import random
import transformers
from docx import Docx
from glob import glob
from tqdm import tqdm


def prepare():
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
        # template = json.load(open(f'template_box.json', encoding='utf-8'))
        template = json.load(open(f'template_box.json', encoding='utf-8'))
        for path in tqdm(files):
            name = path.replace(f'/mnt/ceph/tiku/sjb/', '')[:-5]+'.jpg'
            try:
                text2 = docx.docx2html(path, format=2)
                skip = False
                matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
                if len(matches) == 0:
                    counts['nobox'] += 1
                    continue
                # if len(matches) == 0:
                #     text2 = 'None'
                # else:
                #     text2 = ''
                # boxes = []
                text2 = f'{len(matches)} illustrations: '
                for match in matches:
                    if float(match[3]) < 0.1:
                        skip = True
                        break
                    x = int(float(match[1]) * 1024)
                    y = int(float(match[2]) * 1024)
                    w = int(float(match[3]) * 1024)
                    h = int(float(match[4]) * 1024)
                    # if len(matches) == 1:
                    #     text2 = text2.replace(match[0], f'@[{x}, {y}, {x+w}, {y+h}]@')
                    #     boxes.append(f'@[{x}, {y}, {x+w}, {y+h}]@')
                    # else:
                    #     if random.random() < 0.5:
                    #         text2 = text2.replace(match[0], f'@[{x}, {y}, {x+w}, {y+h}]@')
                    #         boxes.append(f'@[{x}, {y}, {x+w}, {y+h}]@')
                    #     else:
                    #         text2 = text2.replace(match[0], '')
                    text2 += f'@[{x}, {y}, {x+w}, {y+h}]@ '
                    # text2 = text2.replace(match[0], '')
                text2 = text2[:-1] + '.'
                # if len(boxes) == 0:
                #     match = matches[random.randint(0, len(matches)-1)]
                #     text2 = text2.replace(match[0], f'@[{x}, {y}, {x + w}, {y + h}]@')
                #     boxes.append(f'@[{x}, {y}, {x + w}, {y + h}]@')
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
            # for suffix in ['_crop_hw_1226', '_crop_hwn_1226',
            #                '_crop_pp_1213', '_crop_ppn_1213',
            #                '_crop_ppc_1225', '_crop_ppcn_1225']:
            #     if '@[' in text2 and 'n' in suffix:
            #         continue
            for suffix in ['_crop_pp_1213', '_crop_ppc_1225']:
                # text1 = '<image>\n' + random.choice(template)[:-1] + ' with illustrations. The coordinates of the illustrations must be ensured to be very precise. The coordinates are based on 1024x1024 image.'
                # random.shuffle(boxes)
                # text1 = '<image>\n' + random.choice(template) + ' Illustrations: ' + ' '.join(boxes) + '. Do not add other illustrations.'
                text1 = '<image>\n' + random.choice(template)[:-1] + ' with illustrations. The coordinates of the illustrations must be ensured to be very precise. The coordinates are based on 1024x1024 image.'
                text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
                tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
                if len(tokenized.input_ids[0]) > 2048:
                    counts['toolong'] += 1
                    continue
                assert os.path.exists(f'/mnt/ceph/tiku/sjb/{name + suffix + ".jpg"}')
                conversations.append({"image": name + suffix + '.jpg',
                                      "conversations":
                                          [{"from": "human", "value": text1},
                                           {"from": "gpt", "value": text2}]})
                counts['success'] += 1
    print(len(conversations), counts)
    with open(f'/mnt/ceph/tiku/sjb/conversations_box.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


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
            name = path.replace(f'/mnt/ceph/tiku/sjb/', '')[:-5]+'.jpg'
            try:
                text2 = docx.docx2html(path, format=2)
                matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
                skip = False
                for match in matches:
                    if float(match[3]) < 0.1:
                        skip = True
                        break
                    # centery = (float(match[2]) + float(match[4])) / 2
                    text2 = text2.replace(match[0], f'[box]')
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
                text1 = '<image>\n' + random.choice(template)
                text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
                tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
                if len(tokenized.input_ids[0]) > 2048:
                    counts['toolong'] += 1
                    continue
                assert os.path.exists(f'/mnt/ceph/tiku/sjb/{name + suffix + ".jpg"}')
                conversations.append({"image": name + suffix + '.jpg',
                                      "conversations":
                                          [{"from": "human", "value": text1},
                                           {"from": "gpt", "value": text2}]})
                counts['success'] += 1
    print(len(conversations), counts)
    with open(f'/mnt/ceph/tiku/sjb/conversations_html.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def prepare_tiku_html():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

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

    conversations = []
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'toosmall': 0, 'nobox': 0}
    docx = Docx()
    template = json.load(open(f'template_html.json', encoding='utf-8'))
    for path in tqdm(files):
        name = path.replace(f'/mnt/ceph/tiku/', '')
        path = path.replace('/images', '/words')[:-4] + '.docx'
        assert os.path.exists(path)
        try:
            text2 = docx.docx2html(path, format=2)
            matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
            skip = False
            for match in matches:
                if float(match[3]) < 0.1:
                    skip = True
                    break
                text2 = text2.replace(match[0], '[box]')
            if skip:
                counts['toosmall'] += 1
                continue
        except Exception as e:
            print(e)
            counts['error'] += 1
            continue
        text1 = '<image>\n' + random.choice(template)
        text3 = text1.replace('<image>', replace_token) + '\n' + text2 + '\n</s>'
        tokenized = tokenizer(text3, return_tensors="pt", padding="longest", max_length=4096, truncation=True)
        if len(tokenized.input_ids[0]) > 2048:
            counts['toolong'] += 1
            continue
        conversations.append({"image": name,
                              "conversations":
                                  [{"from": "human", "value": text1},
                                   {"from": "gpt", "value": text2}]})
        counts['success'] += 1
    print(len(conversations), counts)
    with open(f'/mnt/ceph/tiku/sjb/conversations_tiku_html.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def check():
    import re
    import cv2
    docx = Docx()
    conversations = json.load(open('/mnt/ceph/tiku/sjb/conversations_html.json', encoding='utf-8'))
    print(len(conversations))
    for c in tqdm(conversations):
        # if '200680218-1' not in c['image']:
        #     continue
        path = os.path.join('/mnt/ceph/tiku/sjb', c['image'])
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
    # data = json.load(open('/mnt/ceph/28/datasets/data_large/conversations-10000.json', encoding='utf-8'))
    # random.shuffle(data)
    # with open('/mnt/ceph/28/datasets/data_large/conversations-2000.json', 'w', encoding='utf-8') as f:
    #     json.dump(data[:2000], f, ensure_ascii=False, indent=2)
