import os
import json
import cv2
import random
import sys
import re
import transformers
import multiprocessing
import rotated_rect_utils
from docx import Docx
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<ref><box>'


def map_obj(boxes_value, boxes_seq):
    """
    >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
    >>> boxes_seq_ = [[3, 1], [2]]
    >>> var = map_obj(normalized_boxes, boxes_seq_)
    >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
    """
    try:
        ret = []
        for boxes in boxes_seq:
            boxes_ret = []
            for box_index in boxes:
                if isinstance(box_index, (list, tuple)):
                    boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                else:
                    boxes_ret.append(boxes_value[box_index])
            ret.append(boxes_ret)
        return ret
    except:
        raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")


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


def put_queue(progress_queue, text1, text2, path, tokenizer):
    source = {"image": path,
              "conversations":
                  [{"from": "human", "value": text1},
                   {"from": "gpt", "value": text2}]}
    input_id = preprocess(source, tokenizer)
    if len(input_id) > 2048:
        progress_queue.put(('toolong', path))
    else:
        progress_queue.put(('success', source))


def process(lines, dataset, tokenizer, progress_queue):
    image_folder = dataset['image_folder']
    for line in lines:
        try:
            item = json.loads(line)
            img_path, boxes, boxes_seq, question, caption = dataset['item_lambda'](item)
            image = Image.open(os.path.join(image_folder, img_path))
            w, h = image.size
            for seq in boxes_seq:
                text = ''
                for k in seq:
                    box = boxes[k]
                    x0 = int(box[0] * 1000 // w)
                    y0 = int(box[1] * 1000 // h)
                    x1 = int(box[2] * 1000 // w)
                    y1 = int(box[3] * 1000 // h)
                    text += f'<box>({x0},{y0}),({x1},{y1})</box>'
                if BOXES_PLACEHOLDER in question:
                    question = question.replace(BOXES_PLACEHOLDER, text, 1)
                else:
                    caption = caption.replace(BOXES_PLACEHOLDER, text, 1)
            put_queue(progress_queue, question, caption, img_path, tokenizer)
        except Exception as e:
            print(e)
            progress_queue.put(('error', str(e)))
            continue


def process_parallel(lines, dataset, tokenizer, progress_queue, output_path):
    chunk_size = len(lines) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, dataset, tokenizer, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(lines))
    counts = {'success': 0, 'error': 0, 'toolong': 0}
    conversations = []
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            if result[0] == 'success':
                conversations.append(result[1])
            progress_bar.set_postfix(count=counts['success'], error=counts['error'],
                                     toolong=counts['toolong'])
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


def prepare_dataset(name, dataset, tokenizer, parallel):
    fileDirname = r'/mnt/ceph/NExT-Chat-main/config/_base_/dataset'
    filename = re.sub(r'{{fileDirname}}', fileDirname, dataset['filename'])
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    lines = lines[:100000]
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(lines, dataset, tokenizer, progress_queue, f'/mnt/ceph2/datasets/tiku/pretrain/conversations_{name}.json')
    else:
        process(lines, dataset, tokenizer, progress_queue)


def prepare():
    tokenizer = transformers.AutoTokenizer.from_pretrained('/mnt/ceph2/pretrained/Ucas/vary-toy/',
                                                           trust_remote_code=True,
                                                           padding_side="right",
                                                           model_max_length=4096)
    print('tokenizer:', tokenizer.__class__.__name__)

    parallel = False if sys.gettrace() is not None else True
    datasets = dict(
        # gc=dict(
        #     type='GCDataset',
        #     filename=r'{{fileDirname}}/../../../data/GC_genome196_train.jsonl',
        #     image_folder=r'/mnt/ceph/datasets/VisualGenome',
        #     template_file=r'{{fileDirname}}/template/GC.json',
        #     item_lambda=lambda item: (item['img_path'], [item['bbox']], [[0]],
        #                               '<image>\nPlease describe the bounding box <ref><box>: ',
        #                               item['expression'] if "expression" in item else item["expressions"][0])
        # ),
        # recvg=dict(
        #     type='RECDataset',
        #     filename=r'{{fileDirname}}/../../../data/GC_genome196_train.jsonl',
        #     image_folder=r'/mnt/ceph/datasets/VisualGenome',
        #     template_file=r'{{fileDirname}}/template/REC.json',
        #     item_lambda=lambda item: (item['img_path'], [item['bbox']], [[0]],
        #                               f'<image> Please find {item["expression"]} in the image: ',
        #                               f'{BOXES_PLACEHOLDER}')
        # ),
        # vqav2=dict(
        #     type='VQAv2Dataset',
        #     filename=r'{{fileDirname}}/../../../data/v2_OpenEnded_mscoco_train2014_questions.jsonl',
        #     image_folder=r'/mnt/ceph/datasets/coco',
        #     template_file=r"{{fileDirname}}/template/VQA.json",
        #     item_lambda=lambda item: (item['image_path'], [], [],
        #                               f"<image>\n{item['question']}",
        #                               f"{item['annotation']['multiple_choice_answer']}.")
        # ),
        caption=dict(
            type='CaptionDataset',
            filename=r'{{fileDirname}}/../../../data/CAP_coco2014_train.jsonl',
            image_folder=r'/mnt/ceph/datasets/coco/train2014',
            template_file=r'{{fileDirname}}/template/image_cap.json',
            item_lambda=lambda item: (item['img_path'], [], [],
                                      "Please describe the image: ",
                                      item['caption'])
        ),
        rec=dict(
            type='RECDataset',
            filename=r'{{fileDirname}}/../../../data/REC_ref3_train.jsonl',
            image_folder=r'/mnt/ceph/datasets/coco/train2014',
            template_file=r'{{fileDirname}}/template/REC.json',
            item_lambda=lambda item: (item['img_path'], [item['bbox']], [[0]],
                                      f'<image> Please find {item["expression"]} in the image: ',
                                      f'{BOXES_PLACEHOLDER}')
        ),
        reg=dict(
            type='REGDataset',
            filename=r'{{fileDirname}}/../../../data/REC_ref3_train.jsonl',
            image_folder=r'/mnt/ceph/datasets/coco/train2014',
            template_file=r'{{fileDirname}}/template/REG.json',
            item_lambda=lambda item: (item['img_path'], [item['bbox']], [[0]],
                                      '<image>\nPlease describe the bounding box <ref><box>: ',
                                      item['expression'] if "expression" in item else item["expressions"][0])
        ),
        # flickr = dict(
        #     type='FlickrDataset',
        #     filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_train.jsonl',
        #     image_folder=r'/mnt/ceph/datasets/flickr30k/flickr30k-images',
        #     template_file=r'{{fileDirname}}/template/flickr30k.json',
        #     item_lambda=lambda item: (f"{item['image_id']}.jpg",
        #                               item['boxes'],
        #                               item['boxes_seq'],
        #                               'Describe the image and give the coordinates of objects:',
        #                               item['sentence'].replace(PHRASE_ST_PLACEHOLDER, "").replace(PHRASE_ED_PLACEHOLDER, BOXES_PLACEHOLDER))
        # )
    )
    name = 'reg'
    prepare_dataset(name, datasets[name], tokenizer, parallel)


def check():
    import re
    conversations = json.load(open('/mnt/ceph2/datasets/tiku/pretrain/conversations_reg.json', encoding='utf-8'))
    dir = r'/mnt/ceph/datasets/coco/train2014'
    # dir = r'/mnt/ceph/datasets/VisualGenome'
    # dir = r'/mnt/ceph/datasets/coco'
    print(len(conversations))
    # for c in tqdm(conversations):
    #     path = dir + c['image']
    #     if not os.path.exists(path):
    #         print(path)
    for c in tqdm(conversations):
        path = os.path.join(dir, c['image'])
        # if '`' not in c['conversations'][1]['value']:
        #     continue
        print(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (1000, 1000))
        for i in range(2):
            text = c['conversations'][i]['value']
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
