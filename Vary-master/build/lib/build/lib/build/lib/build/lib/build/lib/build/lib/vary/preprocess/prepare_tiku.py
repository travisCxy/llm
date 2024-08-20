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

    # files = glob(f'/mnt/ceph/tiku/images_split/**/*.jpg', recursive=True)
    with open('/mnt/ceph/tiku/images_split/files.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    random.shuffle(files)
    files = files[:5000]
    files = [f'/mnt/ceph/tiku/images_split/{file}' for file in files if file]
    conversations = []
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'toosmall': 0, 'nobox': 0}
    docx = Docx()
    # template = json.load(open(f'template_box.json', encoding='utf-8'))
    template = json.load(open(f'template_html.json', encoding='utf-8'))
    for path in tqdm(files):
        name = path.replace('/mnt/ceph/tiku/images_split/', '')
        box_path = path[:-4] + '.json'
        path = path.replace('/images_split/', '/words_split/')[:-4] + '.docx'
        try:
            text2 = docx.docx2html(path, format=2, box_path=box_path)
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
                # text2 += f'@[{x}, {y}, {x+w}, {y+h}]@ '
                # text2 = text2.replace(match[0], '')
                text2 = text2.replace(match[0], f'@[{x}, {y}, {x + w}, {y + h}]@')
            # if len(boxes) == 0:
            #     match = matches[random.randint(0, len(matches)-1)]
            #     text2 = text2.replace(match[0], f'@[{x}, {y}, {x + w}, {y + h}]@')
            #     boxes.append(f'@[{x}, {y}, {x + w}, {y + h}]@')
            if skip:
                counts['toosmall'] += 1
                continue
        except Exception as e:
            print(e)
            counts['error'] += 1
            continue
        text1 = '<image>\n' + random.choice(template)[:-1] + ' with illustrations. The coordinates of the illustrations must be ensured to be very precise. The coordinates are based on 1024x1024 image.'
        # random.shuffle(boxes)
        # text1 = '<image>\n' + random.choice(template) + ' Illustrations: ' + ' '.join(boxes) + '. Do not add other illustrations.'
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
    with open(f'/mnt/ceph/tiku/images_split/conversations_html_box.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def check():
    import re
    import cv2
    docx = Docx()
    conversations = json.load(open('/mnt/ceph/tiku/images_split/conversations_html_box_1.json', encoding='utf-8'))
    for c in tqdm(conversations):
        path = os.path.join('/mnt/ceph/tiku/images_split', c['image'])
        text = c['conversations'][1]['value']
        print(path)
        print(c['conversations'][0]['value'])
        pretty_html = docx.pretty(text, 2, path)
        with open(r'/mnt/ceph/Vary/test/test.html', 'w', encoding='utf-8') as f:
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
