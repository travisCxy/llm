import re
import io
import zipfile
import random
import math
from lxml import etree
from convertor.common import namespaces, fonts


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


def calc_paragraphs_height(paragraphs, font):
    sum = 0
    pageHeight = 445.04
    for paragraph in paragraphs:
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
            # if 'leftChars' in prop:
            #     count -= math.ceil(prop['leftChars'])
            # if 'firstLineChars' in prop:
            #     length -= prop['firstLineChars']
            # if 'hangingChars' in prop:
            #     count -= prop['hangingChars']

        height = math.ceil(length / count)
        if height == 0:
            height = 1
        sum += height
    pitch = pageHeight / font['lines']
    sum = (sum + 1) * pitch
    return sum


def get_sections(body):
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
                continue
            if t.text is None:
                section.append(child)
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
    if len(section) > 0:
        sections.append(section)
    return sections


def create_sectPr(cols_w=[], with_paragraph=True):
    sectPr = etree.Element(f'{{{namespaces["w"]}}}sectPr')
    type = etree.Element(f'{{{namespaces["w"]}}}type')
    type.set(f'{{{namespaces["w"]}}}val', 'continuous')
    pgSz = etree.Element(f'{{{namespaces["w"]}}}pgSz')
    pgSz.set(f'{{{namespaces["w"]}}}w', '11907')
    pgSz.set(f'{{{namespaces["w"]}}}h', '16839')
    pgMar = etree.Element(f'{{{namespaces["w"]}}}pgMar')
    pgMar.set(f'{{{namespaces["w"]}}}top', '900')
    pgMar.set(f'{{{namespaces["w"]}}}right', '1500')
    pgMar.set(f'{{{namespaces["w"]}}}bottom', '900')
    pgMar.set(f'{{{namespaces["w"]}}}left', '1500')
    pgMar.set(f'{{{namespaces["w"]}}}header', '500')
    pgMar.set(f'{{{namespaces["w"]}}}footer', '500')
    pgMar.set(f'{{{namespaces["w"]}}}gutter', '0')
    cols = etree.Element(f'{{{namespaces["w"]}}}cols')
    if len(cols_w) > 1:
        cols.set(f'{{{namespaces["w"]}}}num', str(len(cols_w)))
        cols.set(f'{{{namespaces["w"]}}}space', '0')
        cols.set(f'{{{namespaces["w"]}}}equalWidth', '0')
        for w in cols_w:
            col = etree.Element(f'{{{namespaces["w"]}}}col')
            col.set(f'{{{namespaces["w"]}}}w', str(w))
            col.set(f'{{{namespaces["w"]}}}space', '0')
            cols.append(col)
    else:
        cols.set(f'{{{namespaces["w"]}}}sep', '1')
        cols.set(f'{{{namespaces["w"]}}}space', '425')
    sectPr.append(type)
    sectPr.append(pgSz)
    sectPr.append(pgMar)
    sectPr.append(cols)

    if with_paragraph:
        p = etree.Element(f'{{{namespaces["w"]}}}p')
        pPr = etree.Element(f'{{{namespaces["w"]}}}pPr')
        pPr.append(sectPr)
        p.append(pPr)
        return p
    return sectPr


def save(xml_tree, src, dst):
    modified_xml_content = etree.tostring(xml_tree, encoding='UTF-8', xml_declaration=True)
    output_docx = io.BytesIO()
    with zipfile.ZipFile(src, 'r') as docx_zip, zipfile.ZipFile(output_docx, 'a', zipfile.ZIP_DEFLATED) as new_docx_zip:
        for item in docx_zip.infolist():
            if item.filename == 'word/document.xml':
                content = modified_xml_content
            elif item.filename == 'word/styles.xml':
                content = docx_zip.read(item.filename).decode('utf-8')
                m = re.search(r'w:ascii="Times New Roman" w:eastAsia="(\S+)"', content)
                if m is not None:
                    font = m.group(1)
                    fonts = ['黑体', '微软雅黑', 'MicroftJhengHei', '华文细黑',
                             '华文中宋', '宋体', '等线', '华文宋体', '新宋体', '华文仿宋',
                             '仿宋', '楷体', '华文楷体']
                    if font not in fonts:
                        font = random.choice(fonts)
                        content = content.replace(m.group(0), f'w:ascii="Times New Roman" w:eastAsia="{font}"')
                content = content.encode()
            else:
                content = docx_zip.read(item.filename)
            new_docx_zip.writestr(item, content)
    with open(dst, 'wb') as modified_docx_file:
        modified_docx_file.write(output_docx.getvalue())


def create_clos_for_pic(src, dst):
    with zipfile.ZipFile(src, 'r') as docx_zip:
        with docx_zip.open('word/document.xml', 'r') as document_xml_file:
            xml_content = document_xml_file.read()
        content = docx_zip.read('word/styles.xml').decode('utf-8')
        m = re.search(r'<w:sz\s+w:val="(\d+)"\s*/>', content)
        fontsize = float(m.group(1)) / 2
        font = None
        for f in fonts:
            if fontsize <= f['pt']:
                font = f
                break
    xml_tree = etree.fromstring(xml_content)
    body = xml_tree.find('w:body', namespaces)
    last_sectPr = body.find('.//w:sectPr', namespaces)
    sections = get_sections(body)
    inserted = False
    for section in sections:
        # for i in range(0, len(section)):
        #     child = section[i]
        #     tag = child.tag.split('}')[1]
        #     if tag == 'p':
        #         t = child.find('.//w:t', namespaces)
        #         if t is not None:
        #             print(t.text)
        if len(section) < 1:
            continue
        for i in range(1, len(section)):
            child = section[i]
            tag = child.tag.split('}')[1]
            if tag == 'p':
                drawing = child.find('.//w:drawing', namespaces)
                if drawing is None:
                    r = child.findall('w:r', namespaces)
                    if r is None or len(r) > 1:
                        continue
                    pict = r[0].find('.//w:pict', namespaces)
                    if pict is None:
                        continue
                else:
                    t = child.find('.//w:t', namespaces)
                    if t is not None:
                        continue
                pre_child = section[i-1]
                tag = pre_child.tag.split('}')[1]
                if tag != 'p':
                    continue
                mode = random.choice([0, 1, 2])
                if mode > 0:
                    if i < 3 or sections.index(section) == 0:
                        mode = 0
                if drawing is None:
                    shape = pict.find('.//v:shape', namespaces)
                    if shape is None:
                        continue
                    style = shape.get('style')
                    height = re.search(r'height:(\d+(\.\d+)?)pt', style)
                    if height is None:
                        continue
                    height = float(height.group(1))
                else:
                    extent = drawing.find('.//wp:extent', namespaces)
                    if extent is None:
                        continue
                    height = float(extent.get('cy')) / 914400 * 72
                try:
                    if mode > 0:
                        height2 = calc_paragraphs_height(section[1:i], font)
                    else:
                        height2 = calc_paragraphs_height(section[0:i], font)
                except Exception as e:
                    continue
                if mode == 2:
                    if height2 >= height:
                        mode = 1
                if mode < 2:
                    if height2 >= height * 2:
                        continue
                if mode != 1:
                    for j in range(1, i):
                        pre_child = section[j]
                        tag = pre_child.tag.split('}')[1]
                        if tag != 'p':
                            mode = 1
                            break
                if drawing is not None:
                    extent = drawing.find('.//wp:extent', namespaces)
                    if extent is None:
                        continue
                    cx = int(extent.get('cx'))
                    w = int(cx / 914400 / 6.18 * 8500)
                else:
                    shape = pict.find('.//v:shape', namespaces)
                    if shape is None:
                        continue
                    style = shape.get('style')
                    width = re.search(r'width:(\d+(\.\d+)?)pt', style)
                    if width is None:
                        continue
                    width = float(width.group(1))
                    w = int(width / 445.04 * 8500)
                if w > 3000:
                    continue
                sectPr = create_sectPr()
                if mode == 2:
                    body.remove(child)
                    body.insert(body.index(section[1]), child)
                    body.insert(body.index(child), sectPr)
                    sectPr = create_sectPr([w, 8500-w])
                    body.insert(body.index(pre_child)+1, sectPr)
                else:
                    if mode == 0:
                        body.insert(body.index(section[0]), sectPr)
                    else:
                        body.insert(body.index(section[1]), sectPr)
                    sectPr = create_sectPr([8500-w, w])
                    body.insert(body.index(child)+1, sectPr)
                inserted = True
                break
        # if inserted:
        #     break
    if inserted:
        if last_sectPr is None:
            sectPr = create_sectPr(with_paragraph=False)
            body.append(sectPr)
        else:
            type = etree.Element(f'{{{namespaces["w"]}}}type')
            type.set(f'{{{namespaces["w"]}}}val', 'continuous')
            last_sectPr.append(type)
        save(xml_tree, src, dst)
        return True
    return False


def process_color(src, dst):
    with zipfile.ZipFile(src, 'r') as docx_zip:
        with docx_zip.open('word/document.xml', 'r') as document_xml_file:
            xml_content = document_xml_file.read()
    xml_tree = etree.fromstring(xml_content)
    body = xml_tree.find('w:body', namespaces)
    has_color = False
    for child in body.findall('.//w:p', namespaces):
        pPr = child.find('w:pPr', namespaces)
        if pPr is None:
            continue
        rPr = pPr.find('w:rPr', namespaces)
        if rPr is None:
            run = child.find('w:r', namespaces)
            if run is None:
                continue
            rPr = run.find('w:rPr', namespaces)
            if rPr is None:
                continue
        color = rPr.find('w:color', namespaces)
        if color is None:
            continue
        val = color.get(f'{{{namespaces["w"]}}}val')
        if val == '00B050':
            rPr = pPr.find('w:rPr', namespaces)
            if rPr is not None:
                pPr.remove(rPr)
            for run in child.findall('w:r', namespaces):
                rPr = run.find('w:rPr', namespaces)
                if rPr is not None:
                    run.remove(rPr)
            run = child.find('w:r', namespaces)
            if run is not None:
                t = run.find('w:t', namespaces)
                if t is not None:
                    t.text = t.text.lstrip()
            ind = pPr.find('w:ind', namespaces)
            if ind is None:
                ind = etree.Element(f'{{{namespaces["w"]}}}ind')
                pPr.append(ind)
            else:
                ind.set(f'{{{namespaces["w"]}}}leftChars', '0')
                ind.set(f'{{{namespaces["w"]}}}left', '0')
            ind.set(f'{{{namespaces["w"]}}}firstLineChars', '200')
        elif val == '7030A0':
            rPr = pPr.find('w:rPr', namespaces)
            if rPr is not None:
                pPr.remove(rPr)
            for run in child.findall('w:r', namespaces):
                rPr = run.find('w:rPr', namespaces)
                if rPr is not None:
                    run.remove(rPr)
            jc = pPr.find('w:jc', namespaces)
            if jc is None:
                jc = etree.Element(f'{{{namespaces["w"]}}}jc')
                pPr.append(jc)
            jc.set(f'{{{namespaces["w"]}}}val', 'center')
        else:
            return False
        has_color = True
    if has_color:
        save(xml_tree, src, dst)
    return has_color


if __name__ == '__main__':
    import os
    from glob import glob
    from tqdm import tqdm
    files = glob(r'D:\tiku4\words_split\**\**.docx', recursive=True)
    pbar = tqdm(total=len(files))
    success = 0
    for path in files:
        dst = os.path.join(r'C:\Users\Administrator\Desktop\word\test', os.path.basename(path))
        if create_clos_for_pic(path, dst):
            success += 1
        pbar.update(1)
        pbar.set_postfix(success=success)
    # print(create_clos_for_pic(r'D:\tiku4\words_split\初中\上海\2018-2019学年上海市静安区八年级（下）期末物理试卷_748499_物理_4.docx',
    #                           r'C:\Users\Administrator\Desktop\word\output.docx'))
