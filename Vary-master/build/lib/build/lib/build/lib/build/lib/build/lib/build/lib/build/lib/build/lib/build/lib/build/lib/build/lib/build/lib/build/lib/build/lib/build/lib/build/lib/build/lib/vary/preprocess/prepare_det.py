import cv2
import os
import json
import re
import random
import transformers
import numpy as np
from docx import Docx
from glob import glob
from tqdm import tqdm


def lefttop_rightbottom_theta_to_4points(region):
    x1, y1, x2, y2, theta = region
    points = []
    points.append((x1, y1, 1))
    points.append((x2, y1, 1))
    points.append((x2, y2, 1))
    points.append((x1, y2, 1))

    M = cv2.getRotationMatrix2D((x1, y1), - theta, 1)
    Mt = np.transpose(M)
    roated_points = np.matmul(points, Mt)
    ret = []
    for i in range(roated_points.shape[0]):
        ret.append(tuple(roated_points[i]))
    return ret


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)

    replace_token = '<imgpad>' * 256
    replace_token = '<img>' + replace_token + '</img>'

    files = glob(f'/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/**/*.jpg', recursive=True)
    conversations = []
    counts = {'success': 0, 'error': 0, 'toolong': 0, 'w>h': 0, 'rotation': 0, 'nobox': 0}
    template = json.load(open(f'template_box.json', encoding='utf-8'))
    # for path in tqdm(files):
    #     data = json.load(open(path[:-4] + '.txt', encoding='utf-8'))
    #     regions = data['regions']
    #     skip = False
    #     count = 0
    #     for item in regions:
    #         if abs(item['rotation']) > 1:
    #             skip = True
    #             break
    #         if item['cls'] == 3:
    #             count += 1
    #     if skip:
    #         counts['rotation'] += 1
    #         continue
    #     if count == 0:
    #         counts['nobox'] += 1
    #         continue
    #     counts['success'] += 1
    #     print(counts)
    # print(counts)
    # return

    for path in tqdm(files):
        name = path.replace('/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/', '')
        try:
            data = json.load(open(path[:-4] + '.txt', encoding='utf-8'))
            regions = data['regions']
            skip = False
            count = 0
            for item in regions:
                if abs(item['rotation']) > 1:
                    skip = True
                    break
                if item['cls'] == 3:
                    count += 1
            if skip:
                counts['rotation'] += 1
                continue
            if count == 0:
                counts['nobox'] += 1
                continue

            img = cv2.imread(path)
            h, w, _ = img.shape
            if w > h:
                counts['w>h'] += 1
                continue

            text2 = ''
            for item in regions:
                if item['cls'] != 3:
                    continue
                region = item['region']
                ret = lefttop_rightbottom_theta_to_4points((*region, item['rotation']))
                x1 = min([x for x, y in ret])
                y1 = min([y for x, y in ret])
                x2 = max([x for x, y in ret])
                y2 = max([y for x, y in ret])
                x1 = int(x1 * 1024 / w)
                y1 = int(y1 * 1024 / h)
                x2 = int(x2 * 1024 / w)
                y2 = int(y2 * 1024 / h)
                text2 += f'@[{x1}, {y1}, {x2}, {y2}]@ '
        except Exception as e:
            print(e)
            counts['error'] += 1
            continue
        text1 = '<image>\n' + random.choice(template)[:-1] + ' with illustrations. The coordinates of the illustrations must be ensured to be very precise. The coordinates are based on 1024x1024 image.'
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
        # if len(conversations) >= 10:
        #     break
    print(len(conversations), counts)
    with open(f'/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/conversations_box.json', 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def check():
    docx = Docx()
    conversations = json.load(open('/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/conversations_box.json', encoding='utf-8'))
    for c in tqdm(conversations):
        path = os.path.join('/mnt/ceph/15/datasets/yyt_det/20210602/YYT_DET_20210602/train/', c['image'])
        text = c['conversations'][1]['value']
        print(path)
        print(c['conversations'][0]['value'])
        # pretty_html = docx.pretty(text, 2, path)
        # with open(r'/mnt/ceph/Vary/test/test.html', 'w', encoding='utf-8') as f:
        #     f.write(pretty_html)
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
    prepare()
