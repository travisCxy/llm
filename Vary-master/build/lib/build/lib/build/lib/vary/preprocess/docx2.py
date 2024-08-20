import os
import zipfile
import re
import json
import cv2
from lxml import etree, html
from dwml.omml import oMath2Latex
from pylatexenc.latex2text import LatexNodes2Text
from PIL import Image


class Docx:
    def __init__(self, has_data=True, workdir=None, debug=False):
        self.has_data = has_data
        self.workdir = workdir
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
            rect = descr.split('_')[-4:]
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
                text += '\t'
            elif tag == 'drawing':
                text += self.parse_drawing(child)
            elif tag == 'pict':
                text += self.parse_pict(child)
            elif tag == 'rPr':
                styles += self.parse_run_properties(child, pPr)
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
            # text = f'<span style="{";".join(styles)}">{text}</span>'
            text = f'<span><styles>{";".join(styles)}</styles>{text}</span>'
        return text

    def convert_omml_to_latex(self, math):
        xslt = etree.parse(r'OMML2MML.XSL')
        transform = etree.XSLT(xslt)
        mathml_tree = transform(math)
        xslt =  etree.parse(r'xsltml_2.0/mmltex.xsl')
        transform = etree.XSLT(xslt)
        latex_tree = transform(mathml_tree)
        return str(latex_tree)

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
            text = self.convert_omml_to_latex(math)
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
            # elif tag == 'rFonts':
            #     val = child.get(f'{{{namespaces["w"]}}}ascii')
            #     styles.append(f'font-family:{val}')
            # elif tag == 'sz':
            #     val = child.get(f'{{{namespaces["w"]}}}val')
            #     styles.append(f'font-size:{val}')
            # elif tag == 'color':
            #     val = child.get(f'{{{namespaces["w"]}}}val')
            #     styles.append(f'color:{val}')
            # elif tag == 'i':
            #     styles.append('font-style:italic')
            # elif tag == 'u':
            #     styles.append('text-decoration:underline')
            # elif tag == 'vertAlign':
            #     val = child.get(f'{{{namespaces["w"]}}}val')
            #     if val == 'superscript':
            #         styles.append('vertical-align:super')
            #     elif val == 'subscript':
            #         styles.append('vertical-align:sub')
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
                    styles.append(f'padding-left:{leftChars:.1f}em')
                firstLineChars = child.get(f'{{{self.namespaces["w"]}}}firstLineChars')
                if firstLineChars is not None:
                    firstLineChars = float(firstLineChars) / 100
                    styles.append(f'text-indent:{firstLineChars:.1f}em')
                # hangingChars = child.get(f'{{{self.namespaces["w"]}}}hangingChars')
                # if hangingChars is not None:
                #     hangingChars = float(hangingChars) / 100
                #     styles.append(f'text-hanging:{hangingChars:.1f}em')
        # prop = '' if len(styles) == 0 else f' style="{";".join(styles)}"'
        prop = '' if len(styles) == 0 else f'<styles>{";".join(styles)}</styles>'
        return prop

    def parse_paragraph(self, paragraph):
        text = ''
        prop = ''
        fldChar = None
        math = False
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
                prop = self.parse_paragraph_properties(child)
        if math:
            text += '\\)'
        if self.format > 0:
            text = f'<p>{prop}{text}</p>'
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
                    prop += f'<styles>colspan="{val}"</styles>'
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
            text = f'<td>{prop}{text}</td>'
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
        styles = []
        for child in table:
            tag = child.tag.split('}')[1]
            if tag == 'tr':
                text += self.parse_row(child)
            elif tag == 'tblPr':
                ret = child.find('w:tblBorders', self.namespaces)
                if ret is not None:
                    styles.append('border=1')
                else:
                    ret = child.find('w:tblStyle', self.namespaces)
                    if ret is not None:
                        styles.append('border=1')
                ret = child.find('w:jc', self.namespaces)
                if ret is not None:
                    val = ret.get(f'{{{self.namespaces["w"]}}}val')
                    if val == 'both':
                        val = 'justify'
                    styles.append(f'align={val}')
        if self.format > 0:
            if 'style="text-align:center"' in text:
                text = text.replace(' style="text-align:center"', '')
            text = f'<table><styles>{";".join(styles)}</styles>{text}</table>'
            if len(styles) == 0:
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
        for child in body:
            tag = child.tag.split('}')[1]
            if tag == 'p':
                str += self.parse_paragraph(child)
            elif tag == 'tbl':
                str += self.parse_table(child)
            # elif tag == 'sectPr':
            #     print('Section Properties:', child.tag)
        return str

    def parse_footer(self, tree):
        footer = tree.findall('w:p', self.namespaces)
        str = ''
        for p in footer:
            str += self.parse_paragraph(p)
        return str

    def docx2html(self, docx_path, format, box_path=None, footer=False):
        self.format = format
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
        str = self.parse_body(tree)
        if footer is not None:
            str += self.parse_footer(footer)
        return str

    def pretty(self, body, format=None, img_path=None):
        if format is None:
            format = self.format
        if format > 0:
            pattern = r'(<p><styles>(.*?)</styles>)'
            matches = re.findall(pattern, body)
            for match in matches:
                body = body.replace(match[0], f'<p style="{match[1]}">')
            pattern = r'(<span><styles>(.*?)</styles>)'
            matches = re.findall(pattern, body)
            for match in matches:
                body = body.replace(match[0], f'<span style="{match[1]}">')
            pattern = r'(<table><styles>(.*?)</styles>)'
            matches = re.findall(pattern, body)
            for match in matches:
                body = body.replace(match[0], f'<table {" ".join(match[1].split(";"))}>')
            pattern = r'(<td><styles>(.*?)</styles>)'
            matches = re.findall(pattern, body)
            for match in matches:
                body = body.replace(match[0], f'<td {" ".join(match[1].split(";"))}>')
            print(body)

            body = body.replace('border=1', 'border="1" cellspacing="0" cellpadding="0"')
            # body = body.replace('\t', '\xa0\xa0\xa0\xa0')
            body = body.replace('text-emphasis:dot', 'text-emphasis: dot; text-emphasis-position: under;')
            body = body.replace('<p></p>', '<br>')
            body = body.replace('<pic>', '<img width=100 height=100>')

            if img_path is not None:
                # matches = re.findall(r'(<img x=(\d+) y=(\d+) width=(\d+) height=(\d+)>)', body)
                # if len(matches) > 0:
                #     image = Image.open(img_path)
                #     # dir = os.path.dirname(img_path)
                #     dir = '/mnt/ceph/Vary-toy-main/Vary-master/vary/demo/'
                #     ratio_x = image.width / 1024.0
                #     ratio_y = image.height / 1024.0
                #     for i in range(len(matches)):
                #         match = matches[i]
                #         x0 = int(int(match[1]) * ratio_x)
                #         y0 = int(int(match[2]) * ratio_y)
                #         x1 = int((int(match[1]) + int(match[3])) * ratio_x)
                #         y1 = int((int(match[2]) + int(match[4])) * ratio_y)
                #         crop = image.crop((x0, y0, x1, y1))
                #         crop.save(os.path.join(dir, f'tmp/{i}.png'))
                #         body = body.replace(match[0], f'<img src="tmp/{i}.png" width={match[3]} height={match[4]}>')

                # matches = re.findall(r'(<img x=(\d\.\d+) y=(\d\.\d+) width=(\d\.\d+) height=(\d\.\d+)>)', body)
                # if len(matches) > 0:
                #     image = Image.open(img_path)
                #     # dir = os.path.dirname(img_path)
                #     dir = '/mnt/ceph/Vary-toy-main/Vary-master/vary/demo/'
                #     ratio = image.width / image.height
                #     for i in range(len(matches)):
                #         match = matches[i]
                #         x0 = int(float(match[1]) * image.width)
                #         y0 = int(float(match[2]) * image.height)
                #         x1 = int((float(match[1]) + float(match[3])) * image.width)
                #         y1 = int((float(match[2]) + float(match[4])) * image.height)
                #         w = int(float(match[3]) * 1024 * ratio)
                #         h = int(float(match[4]) * 1024)
                #         crop = image.crop((x0, y0, x1, y1))
                #         crop.save(os.path.join(dir, f'tmp/{i}.png'))
                #         body = body.replace(match[0], f'<img src="tmp/{i}.png" width={w} height={h}>')

                matches = re.findall(r'(<box>\((\d+),(\d+)\),\((\d+),(\d+)\)<\/box>)', body)
                if len(matches) > 0:
                    image = Image.open(img_path)
                    # dir = os.path.dirname(img_path)
                    dir = '/mnt/ceph2/datasets/tiku/vary'
                    # ratio = image.width / image.height
                    for i in range(len(matches)):
                        x0, y0, x1, y1 = map(int, matches[i][1:])
                        x0 = x0 * image.width // 1000
                        y0 = y0 * image.height // 1000
                        x1 = x1 * image.width // 1000
                        y1 = y1 * image.height // 1000
                        crop = image.crop((x0, y0, x1, y1))
                        crop.save(os.path.join(dir, f'tmp/{i}.png'))
                        w = x1 - x0
                        h = y1 - y0
                        body = body.replace(matches[i][0], f'<img src="tmp/{i}.png" width={w} height={h}>')

                # img = cv2.imread(img_path)
                matches = re.findall(r'(<point>\((\d+),(\d+)\)<\/point>)', body)
                # h, w, _ = img.shape
                for match in matches:
                    # x = int(match[1]) * w // 1000
                    # y = int(match[2]) * h // 1000
                    # cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
                    body = body.replace(match[0], '')
                # cv2.imwrite(img_path[:-4] + '-point.jpg', img)


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
        <body>
        ''' + body + '''</body></html>'''
            root = html.fromstring(str)
            # for p in root.xpath('//p'):
            #     if p.text:
            #         p.text = p.text.replace(' ', '\xa0')
            pretty_html = html.tostring(root, encoding='unicode', pretty_print=True)
        else:
            pretty_html = body
        return pretty_html
