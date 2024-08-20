import re
import io
import zipfile
from lxml import etree
from convertor.common import namespaces


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
    if len(section) > 0:
        sections.append(section)
    return sections


def create_sectPr(cols_w=[]):
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

    p = etree.Element(f'{{{namespaces["w"]}}}p')
    pPr = etree.Element(f'{{{namespaces["w"]}}}pPr')
    pPr.append(sectPr)
    p.append(pPr)
    return p


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


def create_clos_for_pic(src, dst):
    with zipfile.ZipFile(src, 'r') as docx_zip:
        with docx_zip.open('word/document.xml', 'r') as document_xml_file:
            xml_content = document_xml_file.read()
    xml_tree = etree.fromstring(xml_content)
    body = xml_tree.find('w:body', namespaces)
    last_sectPr = body.find('.//w:sectPr', namespaces)
    sections = get_sections(body)
    inserted = False
    for section in sections:
        if len(section) < 1:
            continue
        for i in range(1, len(section)):
            child = section[i]
            tag = child.tag.split('}')[1]
            if tag == 'p':
                drawing = child.find('.//w:drawing', namespaces)
                if drawing is None:
                    continue
                t = child.find('.//w:t', namespaces)
                if t is not None:
                    continue
                pre_child = section[i-1]
                tag = pre_child.tag.split('}')[1]
                if tag != 'p':
                    continue
                extent = drawing.find('.//wp:extent', namespaces)
                if extent is None:
                    continue
                cx = int(extent.get('cx'))
                w = int(cx / 914400 / 6.18 * 8820)
                # print(pre_child.find('.//w:t', namespaces).text, w)
                if w > 3000:
                    continue
                sectPr = create_sectPr()
                body.insert(body.index(pre_child), sectPr)
                sectPr = create_sectPr([8820-w, w])
                body.insert(body.index(child)+1, sectPr)
                inserted = True
                break
        # if inserted:
        #     break
    if inserted:
        type = etree.Element(f'{{{namespaces["w"]}}}type')
        type.set(f'{{{namespaces["w"]}}}val', 'continuous')
        last_sectPr.append(type)
        save(xml_tree, src, dst)
        return True
    return False


if __name__ == '__main__':
    # import os
    # from glob import glob
    # from tqdm import tqdm
    # files = glob(r'D:\tiku4\words0\**\**.docx', recursive=True)
    # pbar = tqdm(total=len(files))
    # success = 0
    # for path in files:
    #     dst = os.path.join(r'C:\Users\Administrator\Desktop\word\test', os.path.basename(path))
    #     if create_clos_for_pic(path, dst):
    #         success += 1
    #     pbar.update(1)
    #     pbar.set_postfix(success=success)
    print(create_clos_for_pic(r'D:\tiku4\words0\初中\安徽\机械运动测试题_250058_物理_2.docx',
                              r'C:\Users\Administrator\Desktop\word\output.docx'))
