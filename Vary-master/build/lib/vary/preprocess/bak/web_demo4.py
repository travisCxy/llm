# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""

from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import os
import re
import secrets
import tempfile
import json
import random
import requests

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from vary.utils.conversation import conv_templates, SeparatorStyle
from vary.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from vary.model import *
from vary.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from lxml import html
from PIL import Image
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextIteratorStreamer
from vary.model.plug.transforms import train_transform, test_transform
from docx import Docx
from threading import Thread
import multiprocessing


DEFAULT_CKPT_PATH = '/mnt/ceph2/Vary/runs/0514/'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=5006,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="10.33.10.63",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def _load_model_tokenizer(args):
    print(f"Loading model from {args.checkpoint_path}", flush=True)
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    # model = varyQwenForCausalLM.from_pretrained(args.checkpoint_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True).eval()
    config = varyConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    with init_empty_weights():
        model = varyQwenForCausalLM._from_config(config, torch_dtype=torch.float16)
    no_split_modules = model._no_split_modules
    print(f"no_split_modules: {no_split_modules}", flush=True)
    map_list = {1: "24GB"}
    device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)
    model = load_checkpoint_and_dispatch(model, checkpoint=args.checkpoint_path, device_map=device_map).eval()

    model.to(device='cuda',  dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    return model, tokenizer, image_processor, image_processor_high


def _launch_demo(args, model, tokenizer, image_processor, image_processor_high):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )
    # template = json.load(open(f'template3.json', encoding='utf-8'))

    # triton_client = TritonClient()
    # main_engine = engine.Engine(app.triton_client)
    # dewarp_engine = dewarp_engine.DewarpEngine(app.triton_client)

    def generate(input_ids, kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(input_ids, **kwargs)
            # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # if outputs.endswith(stop_str):
            #     outputs = outputs[:-len(stop_str)]
            # outputs = outputs.strip()

    def get_bbox(region_3point, w, h):
        xs = [region_3point[0], region_3point[2], region_3point[4]]
        ys = [region_3point[1], region_3point[3], region_3point[5]]
        x0 = int(min(xs) * 1000 / w)
        x1 = int(max(xs) * 1000 / w)
        y0 = int(min(ys) * 1000 / h)
        y1 = int(max(ys) * 1000 / h)
        return [x0, y0, x1, y1]

    # def get_ocr(data, w, h):
    #     regions = []
    #     for item in data['regions']:
    #         if item['cls'] not in [1, 10]:
    #             continue
    #         if item['result'][0] == '':
    #             continue
    #         bbox = get_bbox(item['region_3point'], w, h)
    #         regions.append({'bbox': bbox, 'result': item['result'][0]})
    #     pics = [{'bbox': get_bbox(pic['region_3point'], w, h), 'result': f'<pic id="{i}" />'} for i, pic in enumerate(data['pics'])]
    #     regions.extend(pics)
    #     regions = sorted(regions, key=lambda x: x['bbox'][3])
    #     ocr = ''
    #     for region in regions:
    #         ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
    #     return ocr
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
            regions = [region for region in regions if rectangle_overlap_percentage(region['bbox'], bbox) < 80]
        tables = []
        for table in data['tables']:
            rect = table['rect']
            bbox = [int(rect[0] * 1000 / w), int(rect[1] * 1000 / h), int(rect[2] * 1000 / w), int(rect[3] * 1000 / h)]
            cells = ''
            for cell in table['cells']:
                texts = ''
                for text in cell['texts']:
                    if 'content' not in text:
                        continue
                    texts += f"<text>{text['content']}</text>"
                if texts:
                    cells += f'<cell>{texts}</cell>'
            tables.append({'bbox': bbox, 'result': f'<table>{cells}</table>'})
            regions = [region for region in regions if rectangle_overlap_percentage(region['bbox'], bbox) < 80]
            pics = [pic for pic in pics if rectangle_overlap_percentage(pic['bbox'], bbox) < 80]
        if merge:
            print('Merging...')
            merged = True
            while merged:
                for i in range(len(pics)):
                    pic = pics[i]
                    for j in range(i + 1, len(pics)):
                        pic2 = pics[j]
                        center_x = (pic2['bbox'][0] + pic2['bbox'][2]) / 2
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
            print('Merged.')
        regions.extend(pics)
        regions.extend(tables)
        regions = sorted(regions, key=lambda x: x['bbox'][3])
        return regions, pics, tables

    def get_ocr(data, w, h):
        regions, pics, tables = get_regions(data, w, h, merge=True)
        ocr = ''
        for region in regions:
            ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
        return ocr

    def predict(_chatbot, image):
        assert image is not None
        chat_query = _chatbot[-1][0]
        print("User: " + chat_query)

        print("Detecting image...")
        image.save("output/tmp.jpg")
        url = "http://10.33.10.63:5003/image_to_json"
        payload = {}
        files = [('image', ('tmp.jpg', open('output/tmp.jpg', 'rb'), 'image/jpeg'))]
        headers = {}
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        data = json.loads(response.text)
        json.dump(data, open('output/tmp.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        # data = json.load(open('tmp.json', encoding='utf-8'))
        ocr = get_ocr(data, image.size[0], image.size[1])
        print(ocr)
        with open('output/ocr.txt', 'w', encoding='utf-8') as f:
            f.write(ocr)

        message = chat_query
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + f'\nOCR:{ocr}\n' + message
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n' + message
        conv = conv_templates["mpt"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=5)
        image_1 = image.copy()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        image_tensor_1 = image_processor_high(image_1)
        kwargs = dict(images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                      do_sample=False,
                      num_beams=1,
                      # temperature=0.2,
                      streamer=streamer,
                      max_new_tokens=2048,
                      stopping_criteria=[stopping_criteria])
        thread = Thread(target=generate, args=(input_ids, kwargs))
        thread.start()
        # process = multiprocessing.Process(target=generate, args=(input_ids, {}))
        # process.start()

        response = ""
        try:
            for character in streamer:
                response += character
                _chatbot[-1] = (chat_query, response)
                yield _chatbot
        except Exception as e:
            print(e)

        _chatbot[-1] = (chat_query, response)

        print("Qwen-VL-Chat: " + response)
        with open('output/output.txt', 'w', encoding='utf-8') as f:
            f.write(response)
        yield _chatbot

    def regenerate(_chatbot, input_image):
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, input_image)

    def add_text(history, text):
        if text == "":
            # text = random.choice(template)
            text = "Convert with font:"
        history = history + [(text, None)]
        return history

    def add_file(file):
        return file.name

    def reset_user_input():
        return gr.update(value="")

    def reset_state():
        return []

    def convert(history):
        response = history[-1][1]
        docx = Docx()
        pretty_html = docx.pretty(response, 2, 'output/tmp.jpg')
        path = r'output/test.html'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(pretty_html)
        return path

    custom_style = """
    <style>
        p {
            white-space: pre-wrap;
            text-align: justify;
        }
    </style>
    """
    with gr.Blocks(head=custom_style) as demo:
        with gr.Row():
            with gr.Column(scale=4):
                input_image = gr.Image(label="Image", type="pil")
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(label='', elem_classes="control-height", height=750)
        query = gr.Textbox(lines=2, label='Input')
        with gr.Row():
            submit_btn = gr.Button("🚀 Submit (发送)")
            # regen_btn = gr.Button("🤔️ Regenerate (重试)")
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            convert_btn = gr.Button("🧹 Convert (转换)")
        with gr.Row():
            file_download = gr.File(label="Download Link")
        with gr.Row():
            gr.Examples(
                examples=[
                    [
                        os.path.join(os.path.dirname(__file__), "assets/0004_crop_pp.jpg"),
                        "Convert with font:",
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/0220_crop.png"),
                        "Convert with font:",
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/20231220-145333.png"),
                        "Convert with font:",
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/20240305-143839_crop.png"),
                        "Convert with font:",
                    ],
                    [
                        os.path.join(os.path.dirname(__file__), "assets/0253_crop_pp.jpg"),
                        "Convert with font:",
                    ],
                ],
                inputs=[input_image, query],
            )

        submit_btn.click(add_text, [chatbot, query], [chatbot]).then(
            predict, [chatbot, input_image], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [], [chatbot], show_progress=True)
        # regen_btn.click(regenerate, [chatbot, input_image], [chatbot], show_progress=True)
        convert_btn.click(convert, [chatbot], file_download)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer, image_processor, image_processor_high = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer, image_processor, image_processor_high)


if __name__ == '__main__':
    main()
