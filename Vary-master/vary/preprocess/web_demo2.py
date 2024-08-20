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

import os
import requests
import time
from lxml import html
from PIL import Image, ImageDraw
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextIteratorStreamer
from vary.model.plug.transforms import train_transform, test_transform
from threading import Thread
import multiprocessing
from convertor.common import get_regions, sort_regions
from convertor.docx2html import DocxConvertor
from convertor.html2docx import HtmlConvertor


DEFAULT_CKPT_PATH = '/mnt/ceph2/Vary/runs/0819'
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
    # model = varyQwenForCausalLM.from_pretrained(args.checkpoint_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True, load_in_4bit=True)
    # model.load_state_dict(torch.load('/mnt/ceph2/Vary/runs/0711/quantized_model_4bit.pth'))
    # model = varyQwenForCausalLM.from_pretrained(args.checkpoint_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True, load_in_4bit=True).eval()
    # model = varyQwenForCausalLM.from_pretrained(args.checkpoint_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True).eval()
    config = varyConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
    config.pad_token_id = tokenizer.eos_token_id
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    with init_empty_weights():
        model = varyQwenForCausalLM._from_config(config, torch_dtype=torch.bfloat16)
    no_split_modules = model._no_split_modules
    print(f"no_split_modules: {no_split_modules}", flush=True)
    map_list = {0: "24GB"}
    device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)
    print(device_map)
    model = load_checkpoint_and_dispatch(model, checkpoint=args.checkpoint_path, device_map=device_map).eval()

    model.to(device='cuda',  dtype=torch.bfloat16)
    # model = varyQwenForCausalLM.from_pretrained(args.checkpoint_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True).eval()
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    return model, tokenizer, image_processor, image_processor_high


def _launch_demo(args, model, tokenizer, image_processor, image_processor_high):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def generate(input_ids, kwargs):
        start_time = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(input_ids, **kwargs)
        end_time = time.time()
        print(f"Tokens: {output_ids.shape[1] - input_ids.shape[1]}")
        print(f"Time: {end_time - start_time}")
        print(f"Tokens per second: {(output_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time)}")
            # outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # if outputs.endswith(stop_str):
            #     outputs = outputs[:-len(stop_str)]
            # outputs = outputs.strip()

    def get_ocr(data, image):
        w, h = image.size
        regions, pics, tables = get_regions(data, w, h, merge=True)
        for pic in pics:
            pic['result'] = f'<pic id="{pic["id"]}"/>'
        regions.extend(pics)
        regions.extend(tables)
        regions = sort_regions(regions)
        ocr = ''
        draw = ImageDraw.Draw(image)
        for region in regions:
            if 'id' in region:
                ocr += f'<ref id="{region["id"]}">{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
            else:
                ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
            if '<pic' in region['result']:
                x1 = region['bbox'][0] * w // 1000
                y1 = region['bbox'][1] * h // 1000
                x2 = region['bbox'][2] * w // 1000
                y2 = region['bbox'][3] * h // 1000
                # mask = Image.open('1.png')
                # k = (x2 - x1) / (y2 - y1)
                # if k < 2:
                #     mask = mask.crop((0, 0, mask.width // 4, mask.height))
                # elif k < 3:
                #     mask = mask.crop((0, 0, mask.width // 2, mask.height))
                # elif k < 4:
                #     mask = mask.crop((0, 0, mask.width // 4 * 3, mask.height))
                # mask = mask.resize((x2 - x1, y2 - y1))
                # image.paste(mask, (x1, y1, x2, y2))
                draw.rectangle((x1, y1, x2, y2), fill='gray')
                # if (x2 - x1) / (y2 - y1) > 2:
                #     # x = ((x2 - x1) - (y2 - y1) * 2) // 2
                #     # x3 = x1 + x
                #     # x4 = x2 - x
                #     # draw.rectangle((x1, y1, x3, y2), fill='white')
                #     # draw.rectangle((x4, y1, x2, y2), fill='white')
                #     draw.rectangle((x1 + (y2 - y1) * 2, y1, x2, y2), fill='white')
        return ocr

    def detect(path, image):
        url = "http://10.33.10.63:5003/image_to_json"
        payload = {}
        files = [('image', ('tmp.jpg', open(path, 'rb'), 'image/jpeg'))]
        headers = {}
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        data = json.loads(response.text)
        json.dump(data, open('output/tmp.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        # data = json.load(open('tmp.json', encoding='utf-8'))
        ocr = get_ocr(data, image)
        # print(ocr)
        with open('output/ocr.txt', 'w', encoding='utf-8') as f:
            f.write(ocr)
        return ocr

    def predict(_chatbot, image):
        assert image is not None
        chat_query = _chatbot[-1][0]
        print("User: " + chat_query)

        print("Detecting image...")
        path = "output/tmp.jpg"
        image.save(path)
        ocr = detect(path, image)
        # image.save('output/tmp2.jpg')
        # text = ''
        # for t in ocr.split('\n'):
        #     if '<ref>' not in t or '<pic' in t:
        #         text += t + '\n'
        # ocr = text

        message = chat_query
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + f'\nConvert with font:\n'
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + f'\nOCR:\n{ocr}\nConvert options: [font, id].\nstyle: font-family font-size text-align:left center right font-weight:bold \nConvert:'
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + f'\nOCR:\n{ocr}\nConvert options: [font, id].\nConvert:'
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + f'\nOCR and Convert with font:\nOCR:\n{ocr}\nConvert with font:\n'
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n' + message
        # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN
        conv = conv_templates["mpt"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print("Prompt: " + prompt)
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
                      stopping_criteria=[stopping_criteria],
                      prompt_lookup_num_tokens=5,
                      )
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
            print('response:', len(response))
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
            # text = "Convert options: [font].\nConvert:"
            text = 'Convert with font:'
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
        converter = DocxConvertor()
        html = response[response.find('<head>'):]
        html = html.replace('<ref><pic></ref>', '')
        img_path = 'output/tmp.jpg'
        pretty_html = converter.pretty(html, 2, img_path)
        path = 'test.html'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(pretty_html)

        path = 'output.docx'
        with open(r'output/output.txt', 'r', encoding='utf-8') as f:
            html = f.read().replace('\n', '')
        convertor = HtmlConvertor('output')
        convertor.convert(img_path, html, path, 'output/ocr.txt')
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
