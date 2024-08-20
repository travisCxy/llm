import json
import re
from tqdm import tqdm


def convert():
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/lavis/conversations_qms2.json', encoding='utf-8'))
    print(len(conversations))
    conversations2 = []
    for c in tqdm(conversations):
        source = {"image": c['image'],
                  "conversations":
                      [{"from": "human", "value": c['question']},
                       {"from": "gpt", "value": c['caption']}]}
        conversations2.append(source)
    with open('/mnt/ceph2/datasets/tiku/opt/conversations_qms2.json', 'w', encoding='utf-8') as f:
        json.dump(conversations2, f, ensure_ascii=False, indent=2)


def check():
    import cv2
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/opt/conversations_qms2.json', encoding='utf-8'))
    dir = '/mnt/ceph2/datasets/qms/'
    # dir = '/mnt/ceph2/datasets/YYT_DET_20210602/'
    print(len(conversations))
    for c in tqdm(conversations):
        path = dir + c['image']
        print(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        text = c['conversations'][1]['value']
        print(text)
        matches = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', text)
        for match in matches:
            x0, y0, x1, y1 = map(int, match)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    check()
