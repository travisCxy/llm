import os
import zipfile
import re
import json
from lxml import etree, html
from convertor.formula.dwml.omml import oMath2Latex
from pylatexenc.latex2text import LatexNodes2Text
from PIL import Image
from bs4 import BeautifulSoup
from convertor.common import convert_omml_to_latex


class DocxConvertor:
    def __init__(self, has_data=True, workdir=None, with_font=False, debug=False):
        self.has_data = has_data
        self.workdir = workdir
        self.with_font = with_font
        self.debug = debug
        self.namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                           'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
                           'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
                           'r': "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
                           'v': "urn:schemas-microsoft-com:vml"}

    def parse_pict(self, pict):
        assert self.has_data, 'no data'
        assert len(self.data['rects']) > 0, 'no data'
        rect = self.data['rects'].pop(0)
        x = rect[0]
        y = rect[1]
        w = rect[2]
        h = rect[3]
        return f'<img x={x:.3f} y={y:.3f} width={w:.3f} height={h:.3f}>'
        # if self.has_data:
        #     assert len(self.data['rects']) > 0
        #     rect = self.data['rects'].pop(0)
        #     x = rect[0]
        #     y = rect[1]
        #     w = rect[2]
        #     h = rect[3]
        #     return f'<img x={x:.3f} y={y:.3f} width={w:.3f} height={h:.3f}>'
        # else:
        #     shape = pict.find('v:shape', self.namespaces)
        #     style = shape.get('style')
        #     match = re.search(r'width:(\d+)pt;height:(\d+)pt', style)
        #     w, h = int(match[1])*96//72, int(match[2])*96//72
        #     # id = shape.find('v:imagedata', self.namespaces).get('id')
        #     id = shape.get('id')
        #     src = f'tmp/image{id[6:]}.png'
        #     return f'<img width={w} height={h} src="{src}">'


    def parse_drawing(self, drawing):
        if self.has_data:
            assert len(self.data['rects']) > 0, 'no data'
            rect = self.data['rects'].pop(0)
            x, y, w, h = rect
            # x = rect[0] * 1024
            # y = rect[1] * 1024
            # w = rect[2] * 1024
            # h = rect[3] * 1024
            # if x > 5:
            #     x -= 5
            #     w += 10
            # if y > 5:
            #     y -= 5
            #     h += 10
            # x /= 1024
            # y /= 1024
            # w /= 1024
            # h /= 1024
        else:
            docPr = drawing.find('.//wp:docPr', self.namespaces)
            descr = docPr.get('descr')
            if self.debug and descr is None:
                return '<img width=100 height=100 style="background-color:red;">'
            descr = descr.strip()
            rect = descr.split('_')[-4:]
            # if len(rect) != 4:
            #     return '<img width=100 height=100 style="background-color:red;">'
            x = int(rect[0])
            y = int(rect[1])
            w = int(rect[2])
            h = int(rect[3])
            if self.workdir is not None:
                match = re.search(r'_crop_[a-z]+?_\d+?_', descr)
                path = descr[:match.end()-1] + '.txt'
                with open(os.path.join(self.workdir, path), 'a') as f:
                    f.write(f'{float(x)/self.image.width:.3f},{float(y)/self.image.height:.3f},{float(w)/self.image.width:.3f},{float(h)/self.image.height:.3f}\n')
                path = f'tmp/{descr}.png'
                self.image.crop((x, y, x+w, y+h)).save(os.path.join(self.workdir, path))
                x = x * self.ratio_x
                y = y * self.ratio_y
                w = w * self.ratio_x
                h = h * self.ratio_y
                return f'<img src="{path}" x="{int(x)}" y="{int(y)}" width="{int(w)}" height="{int(h)}" />'
            else:
                x = float(x) / self.image.width
                y = float(y) / self.image.height
                w = float(w) / self.image.width
                h = float(h) / self.image.height
        return f'<img x={x:.3f} y={y:.3f} width={w:.3f} height={h:.3f}>'

    def parse_ruby(self, ruby, pPr):
        base = ''
        rt = ''
        for child in ruby:
            tag = child.tag.split('}')[1]
            if tag == 'rubyBase':
                for c in child:
                    tag = c.tag.split('}')[1]
                    if tag == 'r':
                        base += self.parse_run(c, pPr)
            elif tag == 'rt':
                for c in child:
                    tag = c.tag.split('}')[1]
                    if tag == 'r':
                        rt += self.parse_run(c, pPr)
        return f'<ruby>{base}<rt>{rt}</rt></ruby>'

    def parse_run(self, run, pPr):
        text = ''
        styles = []
        for child in run:
            tag = child.tag.split('}')[1]
            if tag == 't':
                if child.text is not None:
                    text += child.text
                pict = child.find('.//w:pict', self.namespaces)
                if pict is not None:
                    text += self.parse_pict(pict)
            elif tag == 'tab':
                text += '<tab/>'
            elif tag == 'drawing':
                text += self.parse_drawing(child)
            elif tag == 'pict':
                text += self.parse_pict(child)
            elif tag == 'rPr':
                for style in self.parse_run_properties(child, pPr):
                    if 'font-size' in style or 'font-family' in style:
                        continue
                    styles.append(style)
            elif tag == 'ruby':
                text += self.parse_ruby(child, pPr)
            elif tag == 'fldChar':
                val = child.get(f'{{{self.namespaces["w"]}}}fldCharType')
                if val == 'begin':
                    return 'fldCharBegin'
                elif val == 'end':
                    return 'fldCharEnd'
            elif tag == 'instrText':
                return child.text
        if 'vertical-align:super' in styles:
            text = f'<sup>{text}</sup>'
            styles.remove('vertical-align:super')
        elif 'vertical-align:sub' in styles:
            text = f'<sub>{text}</sub>'
            styles.remove('vertical-align:sub')
        if text is None:
            text = ''
        elif len(styles) > 0:
            text = f'<span style="{";".join(styles)}">{text}</span>'
        return text

    def parse_math(self, math):
        try:
            omath = oMath2Latex(math)
            text = omath.latex.replace('<', '&lt;').replace('>', '&gt;')
            text = text.replace(r'\left（', '(').replace(r'\right）', ')')

            text = text.replace(r'\rightharpoonaccent', r'\vec')
            if self.debug:
                text = text.replace(r'\~', '\sim')
        except:
            # raise Exception('parse math failed!')
            print('warning: parse math failed!')
            text = convert_omml_to_latex(math)
            if text[0] == '$':
                text = text[1:]
            if text[-1] == '$':
                text = text[:-1]
            text = re.sub(r'\\mathrm{([^}]*)}', r'\1', text)
        # try:
        #     omath = oMath2Latex(math)
        #     text = omath.latex.replace('<', '&lt;').replace('>', '&gt;')
        #     # if bool(re.match(r'^[\(\s\)]*$', text)):
        #     #     return text
        #     # else:
        #     #     return '\\(' + text + '\\)'
        # except:
        #     text = '!!!!!!!!'
        return text

    def parse_run_properties(self, run_properties, pPr):
        styles = []
        for child in run_properties:
            tag = child.tag.split('}')[1]
            if tag == 'b':
                if 'font-weight:bold' not in pPr:
                    styles.append('font-weight:bold')
            elif tag == 'u':
                styles.append('text-decoration:underline')
            elif tag == 'em':
                val = child.get(f'{{{self.namespaces["w"]}}}val')
                if val == 'dot':
                    styles.append('text-emphasis:dot')
            elif tag == 'vertAlign':
                val = child.get(f'{{{self.namespaces["w"]}}}val')
                if val == 'superscript':
                    styles.append('vertical-align:super')
                elif val == 'subscript':
                    styles.append('vertical-align:sub')
            if self.with_font:
                if tag == 'rFonts':
                    val = child.get(f'{{{self.namespaces["w"]}}}eastAsia')
                    if val is not None:
                        styles.append(f'font-family:{val}')
                elif tag == 'sz':
                    val = child.get(f'{{{self.namespaces["w"]}}}val')
                    if val is not None:
                        val = float(val)
                        if val % 2 == 0:
                            styles.append(f'font-size:{int(val)//2}pt')
                        else:
                            styles.append(f'font-size:{val/2}pt')
            # elif tag == 'color':
            #     val = child.get(f'{{{namespaces["w"]}}}val')
            #     styles.append(f'color:{val}')
            # elif tag == 'i':
            #     styles.append('font-style:italic')
        return styles

    def parse_paragraph_properties(self, paragraph_properties):
        styles = []
        for child in paragraph_properties:
            tag = child.tag.split('}')[1]
            if tag == 'jc':
                val = child.get(f'{{{self.namespaces["w"]}}}val')
                if val == 'both':
                    val = 'justify'
                styles.append(f'text-align:{val}')
            elif tag == 'rPr':
                styles += self.parse_run_properties(child, '')
            elif tag == 'ind':
                leftChars = child.get(f'{{{self.namespaces["w"]}}}leftChars')
                if leftChars is not None:
                    leftChars = float(leftChars) / 100
                    if leftChars > 0:
                        styles.append(f'padding-left:{leftChars:.1f}em')
                firstLineChars = child.get(f'{{{self.namespaces["w"]}}}firstLineChars')
                if firstLineChars is not None:
                    firstLineChars = float(firstLineChars) / 100
                    if firstLineChars > 0:
                        styles.append(f'text-indent:{firstLineChars:.1f}em')
                hangingChars = child.get(f'{{{self.namespaces["w"]}}}hangingChars')
                if hangingChars is not None:
                    hangingChars = float(hangingChars) / 100
                    if hangingChars > 0:
                        styles.append(f'text-hanging:{hangingChars:.1f}em')
            elif tag == 'tabs':
                tabs = child.findall('w:tab', self.namespaces)
                if len(tabs) == 1:
                    val = tabs[0].get(f'{{{self.namespaces["w"]}}}val')
                    if val == 'right':
                        styles.append('tab:right')
        return styles

    def parse_paragraph(self, paragraph):
        text = ''
        prop = ''
        fldChar = None
        math = False
        tab_right = False
        for child in paragraph:
            tag = child.tag.split('}')[1]
            if tag == 'r':
                t = self.parse_run(child, prop)
                if t == '':
                    continue
                if t == 'fldCharBegin':
                    assert fldChar is None, 'fldChar is not None'
                    fldChar = ''
                    if self.format == 2 and math:
                        if self.debug:
                            text += f'\\)</span>'
                        else:
                            text += f'\\)'
                        math = False
                elif t == 'fldCharEnd':
                    assert fldChar is not None, 'fldChar is None'
                    match = re.search(r'\\s\\up \d+\(([^)]+)\),([^)]+)', fldChar)
                    if match is not None:
                        base = match.group(2)
                        rt = match.group(1)
                        fldChar = f'<ruby>{base}<rt>{rt}</rt></ruby>'
                    if self.debug:
                        text += f'<span style="color:blue;">{fldChar}</span>'
                    else:
                        text += fldChar
                    fldChar = None
                elif fldChar is not None:
                    fldChar += t
                elif self.format == 2 and math:
                    if self.debug:
                        text += f'\\)</span>{t}'
                    else:
                        text += f'\\){t}'
                    math = False
                else:
                    text += t
            elif tag == 'oMath':
                t = self.parse_math(child)
                if self.debug:
                    text += '<span style="color:red;">'
                if self.format == 2:
                    if math:
                        text += t
                    else:
                        text += f'\\({t}'
                        math = True
                else:
                    l2t = LatexNodes2Text()
                    t = t.replace('&lt;', '<').replace('&gt;', '>')
                    t = l2t.latex_to_text(t)
                    t = t.replace('<', '&lt;').replace('>', '&gt;')
                    text += t
            elif tag == 'oMathPara':
                omath = child.findall('.//m:oMath', self.namespaces)
                assert len(omath) == 1 and math is False, 'oMathPara'
                t = self.parse_math(omath[0])
                if self.debug:
                    text += '<span style="color:red;">'
                if self.format == 2:
                    text += f'\\({t}\\)'
                else:
                    l2t = LatexNodes2Text()
                    t = t.replace('&lt;', '<').replace('&gt;', '>')
                    t = l2t.latex_to_text(t)
                    t = t.replace('<', '&lt;').replace('>', '&gt;')
                    text += t
            elif tag == 'pPr':
                styles = self.parse_paragraph_properties(child)
                for style in styles:
                    if style == 'tab:right':
                        tab_right = True
                        styles.remove(style)
                        break
                prop = '' if len(styles) == 0 else f' style="{";".join(styles)}"'
        # if text and text.replace('_', '') == '':
        #     if len(text) > 70:
        #         text = '<underline/>'
        if tab_right and text.count('<tab/>') == 1:
            text = text.replace('<tab/>', '<tab val="right"/>')
        if math:
            text += '\\)'
        if self.format > 0:
            text = f'<p{prop}>{text}</p>'
            # text = f'<p>{text}</p>'
        else:
            text += '\n'
        return text

    def parse_cell_properties(self, cell_properties):
        prop = ''
        for child in cell_properties:
            tag = child.tag.split('}')[1]
            if tag == 'gridSpan':
                val = child.get(f'{{{self.namespaces["w"]}}}val')
                if val != '1':
                    prop += f' colspan="{val}"'
        return prop

    def parse_cell(self, cell):
        text = ''
        prop = ''
        for child in cell:
            tag = child.tag.split('}')[1]
            if tag == 'p':
                text += self.parse_paragraph(child)
            elif tag == 'tcPr':
                prop = self.parse_cell_properties(child)
        if self.format > 0:
            text = f'<td{prop}>{text}</td>'
        else:
            text += '\t'
        return text

    def parse_row(self, row):
        text = ''
        for child in row:
            tag = child.tag.split('}')[1]
            if tag == 'tc':
                text += self.parse_cell(child)
        if self.format > 0:
            text = f'<tr>{text}</tr>'
        else:
            text += '\n'
        return text

    def parse_table(self, table):
        text = ''
        prop = ''
        for child in table:
            tag = child.tag.split('}')[1]
            if tag == 'tr':
                text += self.parse_row(child)
            elif tag == 'tblPr':
                ret = child.find('w:tblBorders', self.namespaces)
                if ret is not None:
                    prop += ' border="1"'
                else:
                    ret = child.find('w:tblStyle', self.namespaces)
                    if ret is not None:
                        prop += ' border="1"'
                ret = child.find('w:jc', self.namespaces)
                if ret is not None:
                    val = ret.get(f'{{{self.namespaces["w"]}}}val')
                    if val == 'both':
                        val = 'justify'
                    prop += f' align={val}'
        if self.format > 0:
            if 'style="text-align:center"' in text:
                text = text.replace(' style="text-align:center"', '')
            text = f'<table{prop}>{text}</table>'
            if prop == '':
                xml = etree.fromstring(text)
                ps = xml.findall('.//p')
                title = ''
                count = 0
                for p in ps:
                    if p.text is not None:
                        title += etree.tostring(p, encoding='unicode')
                        count += 1
                if count == 1:
                    text = title
            else:
                text = re.sub(r'<td><p>(.*?)</p></td>', r'<td>\1</td>', text)
        else:
            text += '\n'
        return text

    def parse_body(self, tree):
        body = tree.find('w:body', self.namespaces)
        str = ''
        sections = []
        for child in body:
            tag = child.tag.split('}')[1]
            sectPr = None
            if tag == 'p':
                sectPr = child.find('w:pPr/w:sectPr', self.namespaces)
                str += self.parse_paragraph(child)
            elif tag == 'tbl':
                str += self.parse_table(child)
            elif tag == 'sectPr':
                sectPr = child
            if sectPr is not None and str != '':
                cols = sectPr.find('w:cols', self.namespaces)
                if cols is not None:
                    num = cols.get(f'{{{self.namespaces["w"]}}}num')
                    if num is not None and int(num) > 1:
                        assert int(num) == 2, 'cols num'
                        section = None
                        if tag != 'sectPr':
                            try:
                                soup = BeautifulSoup(str, 'html.parser')
                                children_list = list(soup.children)
                                if len(children_list) > 1:
                                    # has_text = False
                                    # for c in children_list:
                                    #     if c.name == 'p' and c.text != '':
                                    #         has_text = True
                                    #         break
                                    last = children_list[-1]
                                    if last.name == 'p':
                                        if last.text == '' and len(last.contents) == 0:
                                            last = children_list[-2]
                                            str = str[:str.rfind('<p')]
                                    i = 0
                                    if last.name == 'p' and (len(last.contents) == 1 and (last.contents[0].name == 'img' or last.contents[0].name == 'table')):
                                        i = str.rfind('<p')
                                    elif last.name == 'table':
                                        i = str.rfind('<table')
                                    if i > 0:
                                        section = '<div style="display: flex"><div style="flex: 1">'+str[:i]+'</div><div>'+str[i:]+'</div></div>'
                                        # if has_text:
                                        #     section = '<div style="display: flex"><div style="flex: 1">'+str[:i]+'</div><div>'+str[i:]+'</div></div>'
                                        # else:
                                        #     section = '<div style="display: flex"><div>'+str[:i]+'</div><div>'+str[i:]+'</div></div>'
                            except:
                                pass
                        str = f'<section cols={num}>{str}</section>' if section is None else section
                sections.append(str)
                str = ''
        str = ''.join(sections) + str
        return str

    def parse_footer(self, tree):
        footer = tree.findall('w:p', self.namespaces)
        str = ''
        for p in footer:
            str += self.parse_paragraph(p)
        return str

    def parse_styles(self, tree):
        default = tree.find('.//w:rPrDefault', self.namespaces)
        rPr = default.find('w:rPr', self.namespaces)
        styles = ''
        for child in rPr:
            tag = child.tag.split('}')[1]
            if tag == 'sz':
                val = child.get(f'{{{self.namespaces["w"]}}}val')
                if val is not None:
                    val = float(val)
                    if val % 2 == 0:
                        styles += f'font-size:{int(val)//2}pt;'
                    else:
                        styles += f'font-size:{val/2}pt;'
            elif tag == 'rFonts':
                val = child.get(f'{{{self.namespaces["w"]}}}eastAsia')
                if val is not None:
                    styles += f'font-family:{val};'
        return styles

    def docx2html(self, docx_path, format, box_path=None, footer=False):
        self.format = format
        str = ''
        with zipfile.ZipFile(docx_path, 'r') as z:
            xml_content = z.read('word/document.xml').replace(b'\xe2\x80\x8b', b'')
            tree = etree.XML(xml_content)
            if footer:
                try:
                    footer = z.read('word/footer1.xml')
                    footer = etree.XML(footer)
                except:
                    footer = None
            else:
                footer = None
            if self.with_font:
                try:
                    styles = z.read('word/styles.xml')
                    styles = etree.XML(styles)
                    styles = self.parse_styles(styles)
                    str = f'<head><style>body{{{styles}}}</style></head>'
                except:
                    pass
        if box_path is None:
            box_path = docx_path[:-5]+'.json'
        if os.path.exists(box_path):
            with open(box_path, 'r', encoding='utf-8') as f:
                rects = json.load(f)
            self.data = {'rects': rects}
        else:
            self.data = {'rects': []}
        if os.path.exists(docx_path[:-5]+'.txt'):
            os.remove(docx_path[:-5]+'.txt')
        if os.path.exists(docx_path[:-5] + '.jpg'):
            self.image = Image.open(docx_path[:-5] + '.jpg')
            if self.workdir is not None:
                self.ratio_x = 1024.0 / self.image.height
            else:
                self.ratio_x = 1024.0 / self.image.width
            self.ratio_y = 1024.0 / self.image.height
        else:
            self.image = None
        str += '<body>'
        str += self.parse_body(tree)
        if footer is not None:
            str += self.parse_footer(footer)
        str += '</body>'
        return str

    def pretty(self, body, format=None, img_path=None):
        if format is None:
            format = self.format
        if format > 0:
            body = body.replace('border="1"', 'border="1" cellspacing="0" cellpadding="0"')
            # body = body.replace('\t', '\xa0\xa0\xa0\xa0')
            body = body.replace('text-emphasis:dot', 'text-emphasis: dot; text-emphasis-position: under;')
            body = body.replace('<p></p>', '<br>')
            body = body.replace('<pic>', '<img width=100 height=100>')

            if img_path is not None:
                matches = re.findall(r'(<pic id="(\d+)"/>)', body)
                if len(matches) > 0:
                    pics = {}
                    with open('output/ocr.txt', 'r') as f:
                        for line in f:
                            match = re.search(r'<pic id="(\d+)"/>', line)
                            if match is not None:
                                id = int(match.group(1))
                                match = re.search(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', line)
                                pics[id] = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                    image = Image.open(img_path)
                    for i in range(len(matches)):
                        id = int(matches[i][1])
                        if id not in pics:
                            # print('pic not exists:', id)
                            continue
                        x0, y0, x1, y1 = pics[id]
                        x0 = x0 * image.width // 1000
                        y0 = y0 * image.height // 1000
                        x1 = x1 * image.width // 1000
                        y1 = y1 * image.height // 1000
                        crop = image.crop((x0, y0, x1, y1))
                        crop.save(f'output/tmp/{id}.png')
                        w = (x1 - x0) * 768 // image.width
                        h = (y1 - y0) * 768 // image.width
                        body = body.replace(matches[i][0], f'<img src="tmp/{id}.png" width={w} height={h}>')

            str = '''<!DOCTYPE html>
        <html>
        <head>
            <script type="text/javascript" async
                src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML">
            </script>
            <style>
                body {
                    margin: 50px;
                }
                p {
                    white-space: pre-wrap;
                    text-align: justify;
                }
                img {
                    border: 2px solid red;
                    border-style: dashed;
                }
                table td {
                    text-align: center;
                }
            </style>
        </head>
        ''' + body + '''</html>'''
            root = html.fromstring(str)
            # for p in root.xpath('//p'):
            #     if p.text:
            #         p.text = p.text.replace(' ', '\xa0')
            pretty_html = html.tostring(root, encoding='unicode', pretty_print=True)
        else:
            pretty_html = body
        return pretty_html


def save_images(docx_path, save_dir):
    with zipfile.ZipFile(docx_path, 'r') as docx:
        for item in docx.infolist():
            if item.filename.startswith('word/media/'):
                filename = os.path.basename(item.filename)
                if filename == '':
                    continue
                save_path = os.path.join(save_dir, filename)
                with open(save_path, 'wb') as f:
                    f.write(docx.read(item.filename))


if __name__ == '__main__':
    workdir = r'C:\Users\Administrator\Desktop\word'
    convertor = DocxConvertor(has_data=False, workdir=workdir, debug=True)
    path = r'C:\Users\Administrator\Desktop\word\output.docx'
    save_images(path, workdir + r'\tmp')
    body = convertor.docx2html(path, format=2)
    pretty_html = convertor.pretty(body)
    with open(path[:-5] + '.html', 'w', encoding='utf-8') as f:
        f.write(pretty_html)
