import os
import random
import multiprocessing
import fitz
import json
import csv
import numpy as np
from glob import glob
from tqdm import tqdm
from spire.doc import Document, FileFormat
from PIL import Image, ImageDraw


def remove_watermark(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            for item in page.get_text("dict")["blocks"]:
                if item["type"] == 0:  # 文本类型
                    for line in item["lines"]:
                        for span in line["spans"]:
                            # if is_red(span["color"]):
                            if span["text"] == "Evaluation Warning: The document was created with Spire.Doc for Python.":
                                rect = fitz.Rect(span["bbox"])
                                page.add_redact_annot(rect, fill=(1, 1, 1))
                                page.apply_redactions()
                                raise StopIteration
    except:
        pass
    doc.save(output_path)


def docx2pdf(docx_path, pdf_path):
    document = Document()
    document.LoadFromFile(docx_path)
    text = document.GetText()
    if len(text.strip().splitlines()) < 3:
        return 'empty', docx_path, 0
    count = document.GetPageCount()
    if count > 1:
        return 'toomanny', docx_path, count
    tmp_path = pdf_path[:-4] + '-wm.pdf'
    document.SaveToFile(tmp_path, FileFormat.PDF)
    document.Close()
    remove_watermark(tmp_path, pdf_path)
    os.remove(tmp_path)
    return 'success', docx_path, document.GetPageCount()


def calc_margin(is_white):
    margin = 0
    for row in is_white:
        if row.all():
            margin += 1
        else:
            break
    return margin


def get_margin(pix):
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    is_white = np.all(img_array >= 250, axis=-1)
    top = calc_margin(is_white)
    bottom = pix.height - calc_margin(reversed(is_white))
    left = calc_margin(is_white.T)
    right = pix.width - calc_margin(reversed(is_white.T))
    return top, bottom, left, right


def pdf2img(path):
    try:
        doc = fitz.open(str(path))
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=random.randint(150, 300))
        top, bottom, left, right = get_margin(pix)
        # print(top, bottom, left, right)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        left = left if left < 10 else random.randint(0, left - 10)
        top = top if top < 10 else random.randint(0, top - 10)
        right = right if right > pix.width - 10 else random.randint(right + 10, pix.width)
        bottom = bottom if bottom > pix.height - 10 else random.randint(bottom + 10, pix.height)
        img = img.crop((left, top, right, bottom))
        img.save(path[:-4] + '.jpg')

        image_info = page.get_image_info()
        if len(image_info) > 0:
            rects = []
            for info in image_info:
                x0, y0, x1, y1 = info['bbox']
                x0 = (x0 / page.rect.width * pix.width - left) / img.width
                y0 = (y0 / page.rect.height * pix.height - top) / img.height
                x1 = (x1 / page.rect.width * pix.width - left) / img.width
                y1 = (y1 / page.rect.height * pix.height - top) / img.height
                rect = [x0, y0, x1 - x0, y1 - y0]
                rects.append(rect)
            json.dump(rects, open(path[:-4] + '.json', 'w', encoding='utf-8'), ensure_ascii=False)

        doc.close()
        return 'success', path, 1
    except Exception as e:
        return str(e), path, 0


def process(chunk, progress_queue):
    for path in chunk:
        # print(path)
        try:
            pdf_path = path.replace('words_split', 'images_split')[:-5] + '.pdf'
            result = docx2pdf(path, pdf_path)
            if result[0] == 'success':
                result = pdf2img(pdf_path)
            progress_queue.put(result)
        except Exception as e:
            progress_queue.put((str(e), path, 0))


def check():
    import cv2
    files = glob(f'/mnt/ceph/tiku/images_split/*.jpg')
    for path in files:
        img = cv2.imread(path)
        img = cv2.resize(img, (1024, 1024))
        if os.path.exists(path[:-4] + '.json'):
            boxes = json.load(open(path[:-4] + '.json', 'r', encoding='utf-8'))
            print(boxes)
            for box in boxes:
                x0, y0, w, h = box
                x1, y1 = x0 + w, y0 + h
                x0, y0, x1, y1 = int(x0 * 1024), int(y0 * 1024), int(x1 * 1024), int(y1 * 1024)
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == 27:
            break


if __name__ == '__main__':
    # check()
    # exit(0)
    # files = glob(f'/mnt/ceph/tiku/words_split/**/*.docx', recursive=True)
    # with open('/mnt/ceph/tiku/words_split/files.txt', 'w', encoding='utf-8') as f:
    #     for file in files:
    #         f.write(file.replace('/mnt/ceph/tiku/words_split/', '') + '\n')
    with open('/mnt/ceph/tiku/words_split/files.txt', 'r', encoding='utf-8') as f:
        files = f.read().split('\n')
    files = [f'/mnt/ceph/tiku/words_split/{file}' for file in files if file]
    # files = glob(f'/mnt/ceph/tiku/words_split/*.docx')
    # random.shuffle(files)
    # for path in tqdm(files):
    #     dir = os.path.dirname(path).replace('words_split', 'images_split')
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
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
    counts = {'success': 0, 'error': 0, 'toomanny': 0, 'empty': 0}
    with open(f'error_docx2pdf.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        while not progress_bar.n >= len(files):
            if all(not runner.is_alive() for runner in runners):
                break
            try:
                result = progress_queue.get(timeout=5)
                if result[0] == 'success':
                    counts['success'] += 1
                    # print(result[1])
                else:
                    counts['error'] += 1
                    if result[0] in counts:
                        counts[result[0]] += 1
                    else:
                        print(result[0])
                        print(result[1])
                    writer.writerow((result[1], result[0]))
                    progress_bar.set_postfix()
                progress_bar.set_postfix(count=counts['success'], error=counts['error'], toomanny=counts['toomanny'], empty=counts['empty'])
            except:
                continue
            progress_bar.update(1)
    progress_bar.close()
    print(counts)
