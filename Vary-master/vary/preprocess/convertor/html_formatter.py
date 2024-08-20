import re
import difflib
import numpy as np
from bs4 import BeautifulSoup
from convertor.common import normalize_text

def format_special_chars(text, latex=False):
    if latex:
        if '\\left\\{' not in text:
            chars = {'○': '○', '〇': '○'}
            for k, v in chars.items():
                text = text.replace(k, f'\\)<span style="font-family:宋体;font-size:22pt">{v}</span>\\(')
            if 'frac' not in text:
                text = text.replace('□', f'\\)<span style="font-family:宋体;font-size:22pt">□</span>\\(')
            chars = {'☆': '☆', '★': '★'}
            for k, v in chars.items():
                text = text.replace(k, f'\\){v}\\(')
    else:
        chars = {'□': '□', '○': '○', '〇': '○'}
        for k, v in chars.items():
            text = text.replace(k, f'<span style="font-family:宋体;font-size:22pt">{v}</span>')
    return text


def average(lst, tolerance=0.2):
    if len(lst) == 0:
        return 0
    mean_height = np.mean(lst)
    std_height = np.std(lst)
    filtered_heights = [h for h in lst if abs(h - mean_height) <= tolerance * std_height]
    if len(filtered_heights) > 0:
        avg_height = np.mean(filtered_heights)
    else:
        avg_height = mean_height
    return avg_height


class HtmlFormatter:
    def __init__(self, refs, pics):
        self.refs = refs
        self.pics = pics

    def align_paragraph(self, topic):
        style = topic['p'].attrs.get('style')
        if style is not None:
            for s in style.split(';'):
                key, value = s.split(':')
                if key == 'padding-left':
                    topic['align'][0] = float(value[:-2])
                elif key == 'text-indent':
                    topic['align'][1] = float(value[:-2])
                elif key == 'text-hanging':
                    topic['align'][2] = float(value[:-2])
        id = topic['id']
        if id is None or len(id) < 2:
            return
        box0 = self.refs[id[0]]['box']
        box1 = self.refs[id[1]]['box']
        if box1[1] > box0[3]:
            if box1[0] - box0[0] > 10:
                if topic['align'][2] == 0:
                    topic['aligned'][2] = 1.7
            elif box0[0] - box1[0] > 10:
                if topic['align'][1] == 0:
                    topic['aligned'][1] = 1.7
            else:
                topic['aligned'][2] = 0
                topic['aligned'][1] = 0

    def align_tree(self, topic_tree):
        for topic in topic_tree['children']:
            if topic['id'] is None or len(topic['children']) == 0:
                continue
            child = topic['children'][0]
            if child['id'] is None or child is None:
                continue
            x0 = min([self.refs[id]['box'][0] for id in topic['id']])
            x1 = min([self.refs[id]['box'][0] for id in child['id']])
            if abs(x0 - x1) < 10:
                if topic['align'][0] > child['align'][0]:
                    topic['aligned'][0] = child['align'][0]
            elif x1 - x0 > 10:
                if topic['align'][0] == child['align'][0]:
                    if topic['parent']['parent'] is None:
                        if topic['align'][0] >= 1.7:
                            topic['aligned'][0] = topic['align'][0] - 1.7
                        elif child['align'][0] == 0:
                            child['aligned'][0] = 1.7
                    else:
                        child['aligned'][0] = topic['align'][0] + 1.7
                elif topic['align'][0] == 0 and topic['align'][1] > 1.7 and child['align'][1] == 0:
                    x0 = self.refs[topic['id'][0]]['box'][0]
                    if x1 - x0 > 10:
                        child['aligned'][0] = topic['align'][1] + 1.7
                elif topic['align'][0] > 0 and topic['align'][1]  == 0 and child['align'][1] == 0:
                    child['aligned'][0] = topic['align'][0] + 1.7
            self.align_tree(topic)

    def align_items(self, items):
        aligned = [None, None, None]
        for t in items:
            for i, a in enumerate(t['aligned']):
                if a is None:
                    continue
                if aligned[i] is None:
                    aligned[i] = a
                elif a != aligned[i]:
                    aligned[i] = False
        if aligned[0] is not None and aligned[1] is not None:
            aligned[0] = 0
        for i, a in enumerate(aligned):
            if a is None or a is False:
                continue
            for t in items:
                t['aligned'][i] = a
        if len(items) > 2 and all(t['aligned'][0] is None for t in items):
            if sum(t['align'][0] > 0 for t in items) == 1:
                for t in items:
                    if t['align'][0] > 0:
                        t['aligned'][0] = 0
                        break

    def apply_align(self, topic_tree):
        for child in topic_tree['children']:
            for i, type in enumerate(['padding-left', 'text-indent', 'text-hanging']):
                if child['aligned'][i] is None:
                    continue
                style = child['p'].attrs.get('style')
                if style is None:
                    style = f'{type}:{child["aligned"][i]}em'
                else:
                    style += f';{type}:{child["aligned"][i]}em'
                child['p'].attrs['style'] = style
            if child['children'] is not None:
                self.apply_align(child)

    def match_id(self, paragraphs):
        keys = [id for id in self.refs.keys() if self.refs[id]['result'] is not None and
                self.refs[id]['result'] != '' and
                '<pic' not in self.refs[id]['result']]
        matched = {id: False for id in keys}
        for paragraph in paragraphs:
            id = paragraph.attrs.get('id')
            if id is None:
                continue
            for i in id.split(';'):
                matched[i] = True

        for paragraph in paragraphs:
            id = paragraph.attrs.get('id')
            if id is None or ';' in id:
                continue
            text = normalize_text(paragraph.text)
            if text is None or text == '':
                continue
            if id not in keys:
                continue
            k = keys.index(id)
            if k == 0 or k == len(keys) - 1:
                continue
            id1 = keys[k - 1]
            id2 = keys[k + 1]
            result = normalize_text(self.refs[id]['result'])
            result1 = normalize_text(self.refs[id1]['result'])
            result2 = normalize_text(self.refs[id2]['result'])
            if result1 is None and result2 is None:
                continue
            if matched[id1] and matched[id2]:
                continue
            ratio = difflib.SequenceMatcher(None, result, text).ratio()
            ratio1 = ratio2 = 0
            if result1 is not None and not matched[id1]:
                ratio1 = difflib.SequenceMatcher(None, result1, text).ratio()
            if result2 is not None and not matched[id2]:
                ratio2 = difflib.SequenceMatcher(None, result2, text).ratio()
            if ratio1 > ratio or ratio2 > ratio:
                paragraph.attrs['id'] = id1 if ratio1 > ratio2 else id2

        k = 0
        for paragraph in paragraphs:
            id = paragraph.attrs.get('id')
            if id is not None:
                continue

            if paragraph.children is not None and len(list(paragraph.children)) == 1:
                child = list(paragraph.children)[0]
                if child.name == 'pic':
                    id = child.attrs.get('id')
                    if id is not None:
                        paragraph.attrs['id'] = id
                    continue

            t1 = normalize_text(paragraph.text)
            if t1 is None or t1 == '':
                continue
            text = ''
            ratio = 0
            ids = []
            for j in range(k, len(keys)):
                i = keys[j]
                result = normalize_text(self.refs[i]['result'])
                if result is None or result == '':
                    continue
                if matched[i]:
                    if text != '':
                        k = j
                        break
                    continue
                ratio2 = difflib.SequenceMatcher(None, text + result, t1).ratio()
                if ratio2 <= ratio:
                    if text != '':
                        k = j
                        break
                    continue
                ratio3 = difflib.SequenceMatcher(None, text + result, t1[:len(text + result)]).ratio()
                if ratio3 < 0.8:
                    if text != '':
                        k = j
                        break
                    continue
                ratio = ratio2
                text += result
                ids.append(i)
                k = j + 1
            if len(ids) == 0:
                continue
            paragraph.attrs['id'] = ';'.join(ids)

    def insert_picture(self, soup, paragraphs):
        pics = []
        for p in paragraphs:
            for c in p.children:
                if c.name == 'pic':
                    id = c.attrs.get('id')
                    if id is not None:
                        pics.extend(id.split(';'))
        for id in self.pics.keys():
            if id in pics:
                continue
            if self.pics[id][2] - self.pics[id][0] < 76 and self.pics[id][3] - self.pics[id][1] < 76:
                continue
            keys = list(self.refs.keys())
            k = keys.index(id)
            if k == 0 or k == len(keys) - 1:
                continue
            id1 = keys[k - 1]
            id2 = keys[k + 1]
            p1 = p2 = None
            children = list(soup.body.children)
            for child in children:
                if child.name == 'p':
                    ids = child.attrs.get('id')
                    if ids is not None:
                        ids = ids.split(';')
                        if id1 in ids:
                            p1 = child
                        elif id2 in ids:
                            p2 = child
                            break
            if p1 is not None and p2 is not None:
                k1 = children.index(p1)
                k2 = children.index(p2)
                if k2 - k1 > 1:
                    children = children[k1 + 1:k2]
                    for child in children:
                        if child.text is not None and child.text.replace(' ', '') != '':
                            child.decompose()
                    p = soup.new_tag('p')
                    pic = soup.new_tag('pic')
                    pic.attrs['id'] = id
                    p.insert(0, pic)
                    p.attrs['id'] = id
                    p1.insert_after(p)

    def align(self, paragraphs):
        for paragraph in paragraphs:
            style = paragraph.attrs.get('style')
            if style is None:
                continue
            id = paragraph.attrs.get('id')
            if id is None or ';' in id:
                continue
            box = self.refs[id]['box']
            if (box[2] - box[0]) / (self.right - self.left) > 0.5:
                continue
            text_align = None
            text = '' if paragraph.text is None else normalize_text(paragraph.text)
            if len(text) > 10 or paragraph.find('pic') is not None:
                for s in style.split(';'):
                    key, value = s.split(':')
                    if key == 'text-align':
                        text_align = value
                        break
                    elif key in ['padding-left', 'text-indent', 'text-hanging']:
                        text_align = 'left'
            if text_align is None or text_align == 'center':
                center1 = (self.left + self.right) / 2
                center2 = (box[0] + box[2]) / 2
                offset = abs(center1 - center2)
                if text_align == 'center':
                    if offset > box[0] - self.left or offset > self.right - box[2]:
                        if box[0] > center1:
                            style = style.replace('text-align:center', 'text-align:right')
                        else:
                            style = style.replace('text-align:center', 'text-align:left')
                else:
                    if box[2] < center1:
                        continue
                    elif box[0] > center1:
                        text_align = 'right'
                    elif offset < box[0] - self.left and offset < self.right - box[2]:
                        w = (self.right - self.left) * 0.2
                        if box[0] - self.left > w and self.right - box[2] > w:
                            text_align = 'center'
                    if text_align is not None:
                        if 'text-align' in style:
                            style = re.sub(r'text-align:[^;]*', f'text-align:{text_align}', style)
                        else:
                            style += f';text-align:{text_align}'
            paragraph.attrs['style'] = style

    def check_tab(self, soup, paragraphs):
        for paragraph in paragraphs:
            if paragraph.children is None:
                continue
            tab_count = len([c for c in paragraph.children if c.name == 'tab'])
            if tab_count != 1:
                continue
            id = paragraph.attrs.get('id')
            if id is None:
                continue
            id = id.split(';')
            if len(id) != 2:
                continue
            box0 = self.refs[id[0]]['box']
            box1 = self.refs[id[1]]['box']
            ratio = (box1[2] - box1[0]) / (box0[2] - box0[0] + box1[2] - box1[0])
            if ratio > 0.75:
                for c in paragraph.children:
                    if c.name == 'tab':
                        c.decompose()
                        p = soup.new_tag('p')
                        count = (box1[0] - box0[2]) // 10
                        if count <= 0:
                            count = 1
                        elif count > 10:
                            count = 10
                        p.string = ' ' * count
                        paragraph.insert(1, p)

    def check_blank(self, soup):
        box = None
        count = 0
        paragraphs = self.get_paragraphs(soup)
        for i in range(len(paragraphs)):
            pic = paragraphs[i].find('pic')
            if pic is not None:
                box = None
                count = 0
                continue
            if paragraphs[i].text is None or paragraphs[i].text.replace(' ', '') == '':
                count += 1
            else:
                ids = paragraphs[i].attrs.get('id')
                if ids is None:
                    box = None
                    count = 0
                    continue
                ids = ids.split(';')
                box2 = [min(self.refs[id]['box'][0] for id in ids),
                        min(self.refs[id]['box'][1] for id in ids),
                        max(self.refs[id]['box'][2] for id in ids),
                        max(self.refs[id]['box'][3] for id in ids)]
                if box is not None:
                    c = int((box2[1] - box[3]) / (self.line_height + self.line_spacing))
                    if c > count:
                        if count < 6:
                            # print('blank', c, count)
                            c = c - count if c - count < 5 else 5
                            for j in range(c):
                                p = soup.new_tag('p')
                                paragraphs[i].insert_before(p)
                    elif c < count:
                        if count > 5:
                            # print('blank', c, count)
                            c = count - c
                            for j in range(i - c, i):
                                paragraphs[j].decompose()
                box = box2
                count = 0

    def get_paragraphs(self, soup):
        paragraphs = []
        for child in soup.body.children:
            if child.name == 'p':
                paragraphs.append(child)
            elif child.name == 'div':
                for c in child.children:
                    if c.name == 'p':
                        paragraphs.append(c)
                    elif c.name == 'div':
                        for c2 in c.children:
                            if c2.name == 'p':
                                paragraphs.append(c2)
        return paragraphs

    def calc(self, paragraphs):
        self.left = 1000
        self.right = 0
        heights = []
        spacings = []
        y = None
        for paragraph in paragraphs:
            id = paragraph.attrs.get('id')
            if id is None:
                y = None
                continue
            id = id.split(';')
            self.left = min(self.left, min(self.refs[i]['box'][0] for i in id))
            self.right = max(self.right, max(self.refs[i]['box'][2] for i in id))
            if paragraph.text is None or paragraph.text == '':
                y = None
                continue
            if '\\frac' in paragraph.text or '\\left' in paragraph.text:
                y = None
                continue
            style = paragraph.attrs.get('style')
            if style is not None and 'font-size' in style:
                y = None
                continue
            boxes = []
            for i in id:
                if len(boxes) == 0:
                    boxes.append(self.refs[i]['box'])
                else:
                    box = self.refs[i]['box']
                    if box[1] > boxes[-1][3]:
                        boxes.append(box)
            heights.extend([box[3] - box[1] for box in boxes])
            spacings.extend([boxes[i][1] - boxes[i - 1][3] for i in range(1, len(boxes))])
            y2 = max([box[3] for box in boxes])
            if y is not None and y2 - y < 50:
                spacings.append(y2 - y)
            y = y2
        avg_height = average(heights)
        avg_spacing = average(spacings)
        if avg_spacing == 0:
            avg_spacing = avg_height
        if self.left == 1000:
            self.left = min([self.refs[i]['box'][0] for i in self.refs.keys()])
        if self.right == 0:
            self.right = max([self.refs[i]['box'][2] for i in self.refs.keys()])
        # print(self.left, self.right)
        # print(heights, avg_height)
        # print(spacings, avg_spacing)
        self.line_height = avg_height
        self.line_spacing = avg_spacing

    def pretty(self, html_str):
        html_str = html_str.replace('\n', '')
        html_str = re.sub(r'(<tab/>\s*)+', r'<tab/>', html_str)

        matches = re.finditer(r'\((.*?)\)', html_str)
        for match in matches:
            if re.match(r'^[ _]*$', match.group(1)):
                html_str = html_str[:match.start() + 1] + ' ' * len(match.group(1)) + html_str[match.end() - 1:]
        matches = re.finditer(r'\(\\\)(.*?)\\\(\)', html_str)
        for match in matches:
            if re.match(r'^[ _]*$', match.group(1)):
                html_str = html_str[:match.start() + 3] + ' ' * len(match.group(1)) + html_str[match.end() - 3:]

        empty_p_pattern = r'(?<!\n)<p(?=\s|>)[^>]*>\s*<\/p>(?!.*\n\s*<p(?=\s|>)[^>]*>\s*<\/p>)'
        matches = [match.span() for match in re.finditer(empty_p_pattern, html_str)]
        if len(matches) == 1:
            html_str = html_str[:matches[0][0]] + html_str[matches[0][1]:]
        elif len(matches) > 1:
            matches.reverse()
            if matches[0][0] != matches[1][1]:
                html_str = html_str[:matches[0][0]] + html_str[matches[0][1]:]
            for i in range(1, len(matches)-1):
                if matches[i][1] != matches[i-1][0] and matches[i][0] != matches[i+1][1]:
                    html_str = html_str[:matches[i][0]] + html_str[matches[i][1]:]
            if matches[-1][1] != matches[-2][0]:
                html_str = html_str[:matches[-1][0]] + html_str[matches[-1][1]:]

        matches = re.finditer(r'\\\(.*?\\\)', html_str)
        i = 0
        str = ''
        if matches is not None:
            for match in matches:
                if i < match.start():
                    text = html_str[i:match.start()]
                    text = format_special_chars(text)
                    str += text
                text = match.group()
                text = format_special_chars(text, True)
                str += text
                i = match.end()
        if i < len(html_str):
            text = html_str[i:]
            text = format_special_chars(text)
            str += text
        html_str = str

        soup = BeautifulSoup(html_str, 'html.parser')
        footer = soup.find('footer')
        if footer is not None:
            footer.decompose()
        paragraphs = self.get_paragraphs(soup)
        self.match_id(paragraphs)
        self.insert_picture(soup, paragraphs)
        paragraphs = self.get_paragraphs(soup)
        paragraphs = [p for p in paragraphs if p.attrs is not None]

        patterns = [r'^\([ ]*\)([0-9]+)', r'^[\(（]?([0-9]+)[\)）]', r'^[例]?([0-9]+)[.、 ]+',
                    r'^[\(（]?([一二三四五六七八九十]+)[.、 \)）]+|([IVXⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)[.、]+']
        topic_tree = {'p': None, 'id': None, 'type': 99, 'order': 0, 'children': [], 'parent': None}
        topic = {'p': None, 'id': None, 'type': -1, 'order': 0, 'children': [], 'parent': topic_tree}
        topics = {}
        for paragraph in paragraphs:
            id = paragraph.attrs.get('id')
            if id is not None:
                id = id.split(';')
            t = {'p': paragraph, 'id': id, 'type': -1, 'order': 0, 'children': [],
                 'align': [0, 0, 0], 'aligned': [None, None, None], 'parent': None}
            self.align_paragraph(t)
            if paragraph.text is not None and paragraph.text != '':
                text = paragraph.text.replace('\\(', '').replace('\\)', '')
                for i, pattern in enumerate(patterns):
                    match = re.match(pattern, text)
                    if match is not None:
                        t['type'] = i
                        t['order'] = match.group(1)
                        topics.setdefault(i, []).append(t)
                        break
            if t['type'] == topic['type']:
                topic = topic['parent']
            elif t['type'] > topic['type']:
                while t['type'] >= topic['type']:
                    topic = topic['parent']
            t['parent'] = topic
            topic['children'].append(t)
            topic = t

        self.align_tree(topic_tree)

        for k, v in topics.items():
            if len(v) < 2:
                continue
            self.align_items(v)
            for t in v:
                if any(len(c['children']) > 0 for c in t['children']):
                    continue
                self.align_items(t['children'])
        self.apply_align(topic_tree)
        self.calc(paragraphs)
        self.align(paragraphs)
        self.check_tab(soup, paragraphs)
        if self.line_height > 0:
            self.check_blank(soup)

        # for child in topic_tree['children']:
        #     print(child['p'].text, child['id'])
        #     for c in child['children']:
        #         print('    ', c['p'].text, c['id'])
        #         for c2 in c['children']:
        #             print('        ', c2['p'].text, c2['id'])
        return soup