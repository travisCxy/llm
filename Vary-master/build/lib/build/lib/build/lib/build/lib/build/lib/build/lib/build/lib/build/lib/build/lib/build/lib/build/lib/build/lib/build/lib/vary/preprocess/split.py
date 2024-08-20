import os
import random
import zipfile
import io
import re
import math
import csv
import multiprocessing
import numpy as np
from glob import glob
from tqdm import tqdm
from lxml import etree, html


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

linePitch = 312 * 25.4 / 1440
pageHeight = 297 - 25.4 * 2
fonts = [{'pt': 9, 'lines': 1.5, 'count': 46},
         {'pt': 10, 'lines': 1.5, 'count': 41},
         {'pt': 10.5, 'lines': 1.5, 'count': 39},
         {'pt': 11, 'lines': 1.5, 'count': 37},
         {'pt': 14, 'lines': 2, 'count': 29},
         {'pt': 15, 'lines': 2, 'count': 27},
         {'pt': 16, 'lines': 2, 'count': 25},
         {'pt': 18, 'lines': 2, 'count': 23}]


def remove_header_footer(sectPr):
    headerReference = sectPr.findall('w:headerReference', namespaces)
    for header in headerReference:
        sectPr.remove(header)
    footerReference = sectPr.findall('w:footerReference', namespaces)
    for footer in footerReference:
        sectPr.remove(footer)


xml_root = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
            'xmlns:cx="http://schemas.microsoft.com/office/drawing/2014/chartex" '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            'xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
            'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" xmlns:w16se="http://schemas.microsoft.com/office/word/2015/wordml/symex" '
            'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
            'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
            'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
            'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
            'mc:Ignorable="w14 w15 w16se wp14">').encode()
xml_root_end = '</w:document>'.encode()


def save(items, document_xml, images, path, font, sz, italic=False):
    output_docx = io.BytesIO()
    sz *= 2
    with zipfile.ZipFile(output_docx, 'a', zipfile.ZIP_DEFLATED) as new_docx_zip:
        for filename, content in items.items():
            if filename == 'word/styles.xml':
                content = content.replace('宋体'.encode(), font.encode())
                content = content.replace('<w:sz w:val="21"/>'.encode(), f'<w:sz w:val="{sz}"/>'.encode())
                if italic:
                    content = content.replace('<w:szCs w:val="22"/>'.encode(),
                                              f'<w:szCs w:val="{sz + 1}"/>\n\t\t\t\t<w:i />'.encode())
                else:
                    content = content.replace('<w:szCs w:val="22"/>'.encode(), f'<w:szCs w:val="{sz + 1}"/>'.encode())
            if filename.startswith('word/media/'):
                if any([image[-12:] in filename for image in images]):
                    new_docx_zip.writestr(filename, content)
            else:
                new_docx_zip.writestr(filename, content)
        document_xml = xml_root + document_xml + xml_root_end
        new_docx_zip.writestr('word/document.xml', document_xml)
    with open(path, 'wb') as modified_docx_file:
        modified_docx_file.write(output_docx.getvalue())


def tbl2para(body):
    for i in range(len(body)):
        child = body[i]
        tag = child.tag.split('}')[1]
        if tag == 'tbl':
            paragraphs = child.findall('.//w:p', namespaces)
            count = 0
            p = None
            for paragraph in paragraphs:
                runs = paragraph.findall('w:r', namespaces)
                if len(runs) > 0:
                    p = paragraph
                    count += 1
            if count == 0:
                body[i] = None
            elif count == 1:
                body[i] = p


def parse_run_properties(run_properties):
    sz = None
    for child in run_properties:
        tag = child.tag.split('}')[1]
        if tag == 'sz':
            sz = float(child.get(f'{{{namespaces["w"]}}}val')) / 2
    return sz


def parse_paragraph_properties( paragraph_properties):
    prop = {}
    for child in paragraph_properties:
        tag = child.tag.split('}')[1]
        if tag == 'rPr':
            fontSize = parse_run_properties(child)
            if fontSize is not None:
                prop['fontSize'] = fontSize
        elif tag == 'ind':
            leftChars = child.get(f'{{{namespaces["w"]}}}leftChars')
            if leftChars is not None:
                prop['leftChars'] = float(leftChars) / 100
            firstLineChars = child.get(f'{{{namespaces["w"]}}}firstLineChars')
            if firstLineChars is not None:
                prop['firstLineChars'] = float(firstLineChars) / 100
            hangingChars = child.get(f'{{{namespaces["w"]}}}hangingChars')
            if hangingChars is not None:
                prop['hangingChars'] = float(hangingChars) / 100
    return prop


def parse_run(run):
    text = ''
    for child in run:
        tag = child.tag.split('}')[1]
        if tag == 't':
            if child.text is not None:
                text += child.text
        elif tag == 'tab':
            text += '\t'
        elif tag == 'drawing':
            raise NotImplementedError('drawing')
        # elif tag == 'rPr':
        #     if child.find('w:sz', namespaces) is not None:
        #         print('ingore run rPr')
        elif tag == 'ruby':
            raise NotImplementedError('ruby')
        elif tag == 'fldChar':
            raise NotImplementedError('fldChar')
        elif tag == 'instrText':
            return child.text
    return text


def parse_paragraph(paragraph):
    text = ''
    prop = None
    for child in paragraph:
        tag = child.tag.split('}')[1]
        if tag == 'r':
            t = parse_run(child)
            if t == '':
                continue
            text += t
        elif tag == 'oMath':
            texts = child.findall('.//m:t', namespaces)
            for t in texts:
                if t.text is not None:
                    text += t.text
        elif tag == 'oMathPara':
            omath = child.findall('.//m:oMath', namespaces)
            assert len(omath) == 1
            texts = omath[0].findall('.//m:t', namespaces)
            for t in texts:
                if t.text is not None:
                    text += t.text
        elif tag == 'pPr':
            prop = parse_paragraph_properties(child)
    return text, prop


def calc_paragraphs_height(paragraph, font):
    picts = paragraph.findall('.//w:pict', namespaces)
    if len(picts) > 0:
        images = []
        height = 0
        for pict in picts:
            shape = pict.find('v:shape', namespaces)
            style = shape.get('style')
            if style is None:
                raise NotImplementedError('v:shape style not found')
            h = int(re.search(r'height:(\d+)pt', style)[1])
            h = h * 25.4 / 72
            if h > height:
                height = h
            id = shape.get('id')
            images.append(id)
        pitch = linePitch * font['lines']
        height = (math.ceil(height / pitch) + 1) * pitch
        return height, images
    text, prop = parse_paragraph(paragraph)
    if prop is None:
        length = len(text)
        count = font['count']
    else:
        if 'fontSize' in prop:
            font = fonts[-1]
            for f in reversed(fonts):
                if prop['fontSize'] < f['pt']:
                    font = f
                else:
                    break
        length = len(text)
        count = font['count']
        if 'leftChars' in prop:
            count -= math.ceil(prop['leftChars'])
        if 'firstLineChars' in prop:
            length -= prop['firstLineChars']
        if 'hangingChars' in prop:
            count -= prop['hangingChars']
    pitch = linePitch * font['lines']
    height = math.ceil(length / count) * pitch
    if height < pitch:
        height = pitch
    return height, []


def calc_table_height(table, font):
    rows = table.findall('.//w:tr', namespaces)
    height = 0
    images = []
    for row in rows:
        cells = row.findall('.//w:tc', namespaces)
        max_height = 0
        for cell in cells:
            paragraphs = cell.findall('.//w:p', namespaces)
            h = 0
            for paragraph in paragraphs:
                ph, imgs = calc_paragraphs_height(paragraph, font)
                if imgs is not None:
                    images.extend(imgs)
                h += ph
            if h > max_height:
                max_height = h
        height += max_height
    return height, images


def split_pages(body, body_tag):
    tbl2para(body)
    pages = []
    page_body = etree.Element(body_tag)
    images = []
    options = [i for i in range(len(fonts))]
    probabilities = [0.1, 0.2, 0.3, 0.2, 0.1, 0.06, 0.03, 0.01]
    k1 = np.random.choice(options, p=probabilities)
    k2 = random.randint(k1, len(fonts) - 1)
    title = True
    sum = 0
    for elem in body:
        if elem is None:
            continue
        tag = elem.tag.split('}')[1]
        imgs = []
        if tag == 'p':
            if title:
                for sz in elem.findall('.//w:sz', namespaces):
                    sz.set(f'{{{namespaces["w"]}}}val', str(fonts[k2]['pt'] * 2))
                for szCs in elem.findall('.//w:szCs', namespaces):
                    szCs.set(f'{{{namespaces["w"]}}}val', str(fonts[k2]['pt'] * 2))
                title = False
            height, imgs = calc_paragraphs_height(elem, fonts[k1])
        elif tag == 'tbl':
            height, imgs = calc_table_height(elem, fonts[k1])
        elif tag not in ['bookmarkStart', 'bookmarkEnd', 'pPr']:
            print(tag)
            height = 0
        if sum + height > pageHeight:
            pages.append((page_body, fonts[k1]['pt'], images))
            page_body = etree.Element(body_tag)
            images = []
            sum = 0
        page_body.append(elem)
        images.extend(imgs)
        sum += height
    pages.append((page_body, fonts[k1]['pt'], images))
    return pages


def split(path):
    with zipfile.ZipFile(path, 'r') as zip_ref:
        items = {}
        for item in zip_ref.infolist():
            if 'word/header' in item.filename or 'word/footer' in item.filename:
                continue
            items[item.filename] = zip_ref.read(item.filename)
        document_xml = items.pop('word/document.xml')
        root = etree.fromstring(document_xml)

        body = root.find('w:body', namespaces)

        for sectPr in body.findall('.//w:sectPr', namespaces):
            remove_header_footer(sectPr)

        k = -1
        for i, elem in enumerate(body):
            if elem.tag.endswith('bookmarkEnd'):
                k = i
                break
        if k == -1:
            return "bookmarkEnd not found", path, 0

        body_tag = body.tag
        body = list(body)
        if not body[k+1].tag.endswith('pPr'):
            return "pPr not found after bookmarkEnd", path, 0
        if body[0].tag.endswith('pict'):
            body.remove(body[0])

        pages = split_pages(body[:k+1], body_tag)

        font = ['宋体', '黑体', '微软雅黑', '楷体', '仿宋', '新宋体',
                 '华文细黑', '华文中宋', '华文仿宋', '华文楷体', '华文宋体'][random.randint(0, 10)]
        path = path[:-5].replace('words', 'words_split')
        # print(path)
        # path = os.path.join('/mnt/ceph/tiku/test', os.path.basename(path))
        # pages = random.sample(pages, 1)
        for i in range(len(pages)):
            save(items, etree.tostring(pages[i][0], encoding='utf-8', method='xml', xml_declaration=False),
                 pages[i][2], path + f'_{i}.docx', font, pages[i][1])
        return 'success', path, len(pages)


def process(chunk, progress_queue):
    for path in chunk:
        try:
            result = split(path)
            progress_queue.put(result)
        except Exception as e:
            progress_queue.put((str(e), path, 0))


if __name__ == '__main__':
    files = glob(f'/mnt/ceph/tiku/words/**/*.docx', recursive=True)
    random.shuffle(files)
    # for path in tqdm(files):
    #     dir = os.path.dirname(path).replace('words', 'words_split')
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
    counts = {'success': 0, 'pages': 0, 'error': 0}
    with open(f'error.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        while not progress_bar.n >= len(files):
            if all(not runner.is_alive() for runner in runners):
                break
            try:
                result = progress_queue.get(timeout=5)
                if result[0] == 'success':
                    counts['success'] += 1
                    counts['pages'] += result[2]
                else:
                    counts['error'] += 1
                    print(result[0])
                    print(result[1])
                    writer.writerow((result[1], result[0]))
                    progress_bar.set_postfix()
                progress_bar.set_postfix(count=counts['success'], error=counts['error'], pages=counts['pages'])
            except:
                continue
            progress_bar.update(1)
    progress_bar.close()
    print(counts)
