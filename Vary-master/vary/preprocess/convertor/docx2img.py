import os
import random
import multiprocessing
import fitz
import json
import sys
import numpy as np
from tqdm import tqdm
from spire.doc import Document, FileFormat
from PIL import Image


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
    if os.path.exists(pdf_path):
        return True
    document = Document()
    document.LoadFromFile(docx_path)
    text = document.GetText()
    if len(text.strip().splitlines()) < 3:
        return False
    count = document.GetPageCount()
    if count > 1:
        # print(docx_path)
        print('page count:', count)
        return False
    tmp_path = pdf_path[:-4] + '-wm.pdf'
    document.SaveToFile(tmp_path, FileFormat.PDF)
    document.Close()
    remove_watermark(tmp_path, pdf_path)
    os.remove(tmp_path)
    return True


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
        return True
    except Exception as e:
        return False


def process(files, progress_queue):
    for docx_path in files:
        pdf_path = docx_path.replace('words', 'images')[:-5] + '.pdf'
        if not docx2pdf(docx_path, pdf_path):
            progress_queue.put(('error', docx_path))
            continue
        if pdf2img(pdf_path):
            progress_queue.put(('success', docx_path))
        else:
            progress_queue.put(('error', docx_path))


def process_parallel(files, progress_queue):
    if len(files) < multiprocessing.cpu_count():
        process(files, progress_queue)
        return
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0}
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            progress_bar.set_postfix(count=counts['success'], error=counts['error'])
        except Exception as e:
            if all(not runner.is_alive() for runner in runners):
                break
            continue
        progress_bar.update(1)
    progress_bar.close()


def docx2img(files):
    parallel = False if sys.gettrace() is not None else True
    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, progress_queue)
    else:
        process(files, progress_queue)
