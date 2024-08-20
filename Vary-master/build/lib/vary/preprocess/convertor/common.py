import os
import latex2mathml.converter
from lxml import etree


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

fonts = [{'pt': 9, 'lines': 32, 'count': 49, 'line_height': 9547200/32/2},
         {'pt': 10, 'lines': 32, 'count': 44, 'line_height': 9547200/32/2},
         {'pt': 10.5, 'lines': 32, 'count': 42, 'line_height': 9547200/32/2},
         {'pt': 11, 'lines': 24, 'count': 40, 'line_height': 9547200/24/2},
         {'pt': 12, 'lines': 24, 'count': 37, 'line_height': 9547200/24/2},
         {'pt': 14, 'lines': 24, 'count': 31, 'line_height': 9547200/24/2},
         {'pt': 15, 'lines': 24, 'count': 29, 'line_height': 9547200/24/2},
         {'pt': 16, 'lines': 24, 'count': 27, 'line_height': 9547200/24},
         {'pt': 18, 'lines': 24, 'count': 24, 'line_height': 9547200/24}]


def convert_latex_to_omml(text):
    mathml = etree.fromstring(latex2mathml.converter.convert(text))
    xslt = etree.parse(os.path.join(os.path.dirname(__file__), 'formula/MML2OMML.XSL'))
    transform = etree.XSLT(xslt)
    omml = transform(mathml)
    omml_tree = etree.fromstring(str(omml))
    return omml_tree


def convert_omml_to_latex(text):
    xslt = etree.parse(os.path.join(os.path.dirname(__file__), 'formula/OMML2MML.XSL'))
    transform = etree.XSLT(xslt)
    mathml_tree = transform(text)
    xslt = etree.parse(os.path.join(os.path.dirname(__file__), 'xsltml_2.0/mmltex.xsl'))
    transform = etree.XSLT(xslt)
    latex_tree = transform(mathml_tree)
    return str(latex_tree)


def get_bbox(region_3point, w, h):
    xs = [region_3point[0], region_3point[2], region_3point[4]]
    ys = [region_3point[1], region_3point[3], region_3point[5]]
    x0 = int(min(xs) * 1000 / w)
    x1 = int(max(xs) * 1000 / w)
    y0 = int(min(ys) * 1000 / h)
    y1 = int(max(ys) * 1000 / h)
    return [x0, y0, x1, y1]


def rectangle_overlap_percentage(rect1, rect2):
    """
    计算第一个矩形在第二个矩形内的百分比。
    矩形格式：[x0, y0, x1, y1]，其中 (x0, y0) 是左上角坐标，(x1, y1) 是右下角坐标。
    """
    ix0 = max(rect1[0], rect2[0])
    iy0 = max(rect1[1], rect2[1])
    ix1 = min(rect1[2], rect2[2])
    iy1 = min(rect1[3], rect2[3])
    if ix1 >= ix0 and iy1 >= iy0:
        intersection_area = (ix1 - ix0) * (iy1 - iy0)
    else:
        intersection_area = 0
    rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
    if rect1_area > 0:
        percentage = (intersection_area / rect1_area) * 100
    else:
        percentage = 0
    return percentage


def convert_table_to_html(cells):
    rows, cols = 0, 0
    for cell in cells:
        rows = max(rows, cell['endrow'] + 1)
        cols = max(cols, cell['endcol'] + 1)
    table_data = [[('', 1, 1) for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        row = cell['startrow']
        col = cell['startcol']
        content = ''
        for text in cell['texts']:
            if text['cls'] not in [1, 10]:
                continue
            if content:
                content += '<br>'
            content += text['content']
        rowspan = cell.get('endrow', row) - row + 1
        colspan = cell.get('endcol', col) - col + 1
        for i in range(row, row + rowspan):
            for j in range(col, col + colspan):
                if i == row and j == col:
                    table_data[i][j] = (content, rowspan, colspan)
                else:
                    table_data[i][j] = (None, 0, 0)

    html = '<table>'
    for row in table_data:
        html += '<tr>'
        for cell_content, rowspan, colspan in row:
            if cell_content is not None:
                if rowspan == 1 and colspan == 1:
                    html += f'<td>{cell_content}</td>'
                elif rowspan > 1 and colspan > 1:
                    html += f'<td rowspan={rowspan} colspan={colspan}>{cell_content}</td>'
                elif rowspan > 1:
                    html += f'<td rowspan={rowspan}>{cell_content}</td>'
                else:
                    assert colspan > 1
                    html += f'<td colspan={colspan}>{cell_content}</td>'

        html += '</tr>'
    html += '</table>'
    return html


def get_regions(data, w, h, merge=False):
    regions = []
    for item in data['regions']:
        if item['cls'] not in [1, 10]:
            continue
        if item['result'][0] == '':
            continue
        bbox = get_bbox(item['region_3point'], w, h)
        regions.append({'bbox': bbox, 'result': item['result'][0]})
    pics = []
    for i, pic in enumerate(data['pics']):
        bbox = get_bbox(pic['region_3point'], w, h)
        pics.append({'bbox': bbox, 'result': f'<pic id="{i}"/>'})
        regions = [region for region in regions if rectangle_overlap_percentage(region['bbox'], bbox) < 70]
    tables = []
    for table in data['tables']:
        rect = table['rect']
        bbox = [int(rect[0] * 1000 / w), int(rect[1] * 1000 / h), int(rect[2] * 1000 / w), int(rect[3] * 1000 / h)]
        table = convert_table_to_html(table['cells'])
        tables.append({'bbox': bbox, 'result': table})
        regions = [region for region in regions if rectangle_overlap_percentage(region['bbox'], bbox) < 70]
        pics = [pic for pic in pics if rectangle_overlap_percentage(pic['bbox'], bbox) < 70]
    if merge:
        merged = True
        while merged:
            for i in range(len(pics)):
                pic = pics[i]
                for j in range(i + 1, len(pics)):
                    pic2 = pics[j]
                    # center_x = (pic2['bbox'][0] + pic2['bbox'][2]) / 2
                    center_y = (pic2['bbox'][1] + pic2['bbox'][3]) / 2
                    # if (center_x < pic['bbox'][0] or center_x > pic['bbox'][2]) and (center_y < pic['bbox'][1] or center_y > pic['bbox'][3]):
                    if center_y < pic['bbox'][1] or center_y > pic['bbox'][3]:
                        merged = False
                        continue
                    x1 = min(pic['bbox'][0], pic2['bbox'][0])
                    y1 = min(pic['bbox'][1], pic2['bbox'][1])
                    x2 = max(pic['bbox'][2], pic2['bbox'][2])
                    y2 = max(pic['bbox'][3], pic2['bbox'][3])
                    bbox = [x1, y1, x2, y2]
                    merged = True
                    for region in regions + tables:
                        if rectangle_overlap_percentage(region['bbox'], bbox) >= 20:
                            merged = False
                            break
                    if merged:
                        pics[i]['bbox'] = bbox
                        pics.remove(pic2)
                        break
                if merged:
                    break
            if len(pics) < 2:
                break
    regions.extend(pics)
    regions.extend(tables)
    regions = sorted(regions, key=lambda x: x['bbox'][3])
    return regions, pics, tables


def match_pics(text2, pics):
    matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', text2)
    used = [False] * len(pics)
    unmatch = False
    for match in matches:
        x0 = int(float(match[1]) * 1000)
        y0 = int(float(match[2]) * 1000)
        x1 = int((float(match[1]) + float(match[3])) * 1000)
        y1 = int((float(match[2]) + float(match[4])) * 1000)
        imgs = []
        ymin = ymax = 0
        for i, pic in enumerate(pics):
            percentage = rectangle_overlap_percentage(pic['bbox'], [x0, y0, x1, y1])
            if percentage < 80:
                continue
            if used[i]:
                unmatch = True
                break
            imgs.append(pic)
            if ymax == 0:
                ymin = pic['bbox'][1]
                ymax = pic['bbox'][3]
            else:
                center_y = (pic['bbox'][1] + pic['bbox'][3]) / 2
                if center_y < ymin or center_y > ymax:
                    unmatch = True
                    break
            used[i] = True
        if len(imgs) == 0:
            unmatch = True
        if unmatch:
            break
        imgs = sorted(imgs, key=lambda x: x['bbox'][0])
        text = ''
        for img in imgs:
            text += img['result']
        text2 = text2.replace(match[0], text)
    if unmatch:
        return None
    return text2