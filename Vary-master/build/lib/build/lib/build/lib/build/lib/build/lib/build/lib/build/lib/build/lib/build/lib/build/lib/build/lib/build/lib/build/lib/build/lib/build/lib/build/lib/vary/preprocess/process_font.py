import os
import random
import zipfile
import re
import io
import fitz
import multiprocessing
import json
import sys
import numpy as np
from glob import glob
from lxml import etree
from tqdm import tqdm
from spire.doc import Document, FileFormat
from PIL import Image


namespaces = {
    'wpc': "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas",
    'cx': "http://schemas.microsoft.com/office/drawing/2014/chartex",
    'mc': "http://schemas.openxmlformats.org/markup-compatibility/2006",
    'o': "urn:schemas-microsoft-com:office:office",
    'r': "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    'm': "http://schemas.openxmlformats.org/officeDocument/2006/math",
    'v': "urn:schemas-microsoft-com:vml",
    'wp14': "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
    'wp': "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    'w10': "urn:schemas-microsoft-com:office:word",
    'w': "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    'w14': "http://schemas.microsoft.com/office/word/2010/wordml",
    'w15': "http://schemas.microsoft.com/office/word/2012/wordml",
    'w16se': "http://schemas.microsoft.com/office/word/2015/wordml/symex",
    'wpg': "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    'wpi': "http://schemas.microsoft.com/office/word/2010/wordprocessingInk",
    'wne': "http://schemas.microsoft.com/office/word/2006/wordml",
    'wps': "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
}
for prefix, uri in namespaces.items():
    etree.register_namespace(prefix, uri)


def get_sections1(body):
    sections = []
    section = []
    index = None
    for child in body:
        tag = child.tag.split('}')[1]
        if tag == 'p':
            run = child.find('w:r', namespaces)
            if run is None:
                continue
            drawing = run.find('w:drawing', namespaces)
            if drawing is not None:
                section.append(child)
                continue
            t = run.find('w:t', namespaces)
            if t is None:
                section.append(child)
                continue
            if t.text is None:
                section.append(child)
                # pict = t.find('.//w:pict', namespaces)
                # if pict is not None:
                #     section.append(child)
                continue
            m = re.match(r'^(\d+)\. ', t.text)
            if m is None:
                section.append(child)
                continue
            i = int(m.group(1))
            if index is None:
                index = i
                if len(section) > 0:
                    sections.append(section)
                section = [child]
            elif i == index + 1:
                index = i
                sections.append(section)
                section = [child]
            else:
                break
        else:
            if len(section) > 0:
                section.append(child)
    return sections


def get_sections2(body):
    sections = []
    for p in body.findall('.//w:p', namespaces):
        run = p.find('w:r', namespaces)
        if run is None:
            continue
        t = run.find('w:t', namespaces)
        if t is None or t.text is None:
            continue
        sections.append([p])
    if len(sections) * 0.2 > 0:
        sections = random.sample(sections, int(len(sections) * 0.2))
    else:
        sections = []
    return sections


def save(xml_tree, src, dst):
    modified_xml_content = etree.tostring(xml_tree, encoding='UTF-8', xml_declaration=True)
    output_docx = io.BytesIO()
    with zipfile.ZipFile(src, 'r') as docx_zip, zipfile.ZipFile(output_docx, 'a', zipfile.ZIP_DEFLATED) as new_docx_zip:
        for item in docx_zip.infolist():
            if item.filename == 'word/document.xml':
                content = modified_xml_content
            else:
                content = docx_zip.read(item.filename)
            new_docx_zip.writestr(item, content)
    with open(dst, 'wb') as modified_docx_file:
        modified_docx_file.write(output_docx.getvalue())


def create_rPr(rPr, font, val):
    if rPr is None:
        rPr = etree.Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rPr", nsmap=namespaces)
    rFonts = rPr.find('w:rFonts', namespaces)
    if rFonts is None:
        rFonts = etree.Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts", nsmap=namespaces)
        rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}ascii", font)
        rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", font)
        rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}hint", "eastAsia")
        rPr.append(rFonts)
    sz = rPr.find('w:sz', namespaces)
    if sz is None:
        sz = etree.Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}sz", nsmap=namespaces)
        sz.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", val)
        rPr.append(sz)
    szCs = rPr.find('w:szCs', namespaces)
    if szCs is None:
        szCs = etree.Element("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}szCs", nsmap=namespaces)
        szCs.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", val)
        rPr.append(szCs)
    return rPr


def process_font(src, dst):
    with zipfile.ZipFile(src, 'r') as docx_zip:
        with docx_zip.open('word/document.xml', 'r') as document_xml_file:
            xml_content = document_xml_file.read()
    xml_tree = etree.fromstring(xml_content)
    body = xml_tree.find('w:body', namespaces)
    # sectPr = body.find('w:sectPr', namespaces)
    # if sectPr is None:
    #     print('sectPr is None')

    if random.random() < 0.5:
        sections = get_sections1(body)
        if len(sections) > 1:
            sections = random.sample(sections, 1)
        else:
            sections = get_sections2(body)
            if int(len(sections) * 0.2) > 0:
                sections = random.sample(sections, int(len(sections) * 0.2))
            else:
                sections = []
    else:
        sections = get_sections2(body)
        if int(len(sections) * 0.2) > 0:
            sections = random.sample(sections, int(len(sections) * 0.2))
        else:
            sections = get_sections1(body)
            if len(sections) > 1:
                sections = random.sample(sections, 1)
            else:
                sections = []
    if len(sections) == 0:
        return False

    for section in sections:
        font = random.choice(['黑体', '微软雅黑', 'MicroftJhengHei', '华文细黑',
                              '华文中宋', '宋体', '等线', '华文宋体', '新宋体', '华文仿宋',
                              '仿宋', '楷体', '华文楷体',
                              '方正姚体', '方正舒体', '华文行楷', '华文新魏', '华文琥珀', '华文隶书'])
        val = str(random.choice([9, 10, 10.5, 11, 12, 14, 15, 16, 18, 20]) * 2)
        for child in section:
            tag = child.tag.split('}')[1]
            if tag != 'p':
                continue
            pPr = child.find('w:pPr', namespaces)
            if pPr is None:
                continue
            pPr.insert(0, create_rPr(None, font, val))
            rPrs = child.findall('.//w:rPr', namespaces)
            for rPr in rPrs:
                create_rPr(rPr, font, val)
            runs = child.findall('.//w:r', namespaces)
            runs += child.findall('.//m:r', namespaces)
            for run in runs:
                rPr = run.find('w:rPr', namespaces)
                if rPr is None:
                    rPr = create_rPr(None, font, val)
                    run.insert(0, rPr)

    save(xml_tree, src, dst)
    return True


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
    count = document.GetPageCount()
    if count > 1:
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
        return 'success', path
    except Exception as e:
        return 'error', path


def process(files, progress_queue):
    for path in files:
        # print(path)
        try:
            docx_path = path.replace('/tiku2', '/tiku3')
            pdf_path = docx_path.replace('/words', '/images')[:-5] + '.pdf'
            retry = 10
            while retry > 0:
                if not process_font(path, docx_path):
                    retry = 0
                    break
                if docx2pdf(docx_path, pdf_path):
                    break
                retry -= 1
            if retry == 0:
                if os.path.exists(docx_path):
                    os.remove(docx_path)
                progress_queue.put(('failed', path))
            else:
                result = pdf2img(pdf_path)
                progress_queue.put(result)
                # print(result)
        except Exception as e:
            print(e)
            progress_queue.put(('error', path))
            continue


def process_parallel(files, progress_queue):
    chunk_size = len(files) // multiprocessing.cpu_count()
    print('cpu count:', multiprocessing.cpu_count())
    file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    runners = [multiprocessing.Process(target=process, args=(chunk, progress_queue)) for chunk in
               file_chunks]
    for runner in runners:
        runner.start()
    progress_bar = tqdm(total=len(files))
    counts = {'success': 0, 'error': 0, 'failed': 0}
    while True:
        try:
            result = progress_queue.get(timeout=5)
            counts[result[0]] += 1
            progress_bar.set_postfix(count=counts['success'], error=counts['error'], failed=counts['failed'])
        except Exception as e:
            if all(not runner.is_alive() for runner in runners):
                break
            continue
        progress_bar.update(1)
    progress_bar.close()
    print(counts)


def process_tiku(parallel):
    files = glob('/mnt/ceph2/datasets/tiku2/**/*.docx', recursive=True)
    # for path in files:
    #     dst = path.replace('/tiku2', '/tiku3')
    #     dir = os.path.dirname(dst)
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)

    progress_queue = multiprocessing.Queue()
    if parallel:
        process_parallel(files, progress_queue)
    else:
        process(files, progress_queue)


if __name__ == '__main__':
    parallel = False if sys.gettrace() is not None else True
    process_tiku(parallel)
