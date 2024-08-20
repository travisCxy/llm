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


DEFAULT_CKPT_PATH = '/mnt/ceph/Vary/runs/0328'
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
    parser.add_argument("--server-port", type=int, default=6006,
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
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/ceph/Vary/runs/0327', trust_remote_code=True)
    model = varyQwenForCausalLM.from_pretrained(args.checkpoint_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True).eval()
    model.to(device='cuda',  dtype=torch.bfloat16)
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    return model, tokenizer, image_processor, image_processor_high


def _launch_demo(args, model, tokenizer, image_processor, image_processor_high):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )
    template = json.load(open(f'template3.json', encoding='utf-8'))

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

    def predict(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        print("User: " + query)
        history_cp = copy.deepcopy(task_history)
        full_response = ""

        image = None
        message = ""
        for q, a in reversed(history_cp):
            if isinstance(q, (tuple, list)):
                if image is None:
                    image = q[0]
            else:
                message = q
            if image is not None and message:
                break
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n' + message
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
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        image = _load_image(image)
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

        response = ""
        for character in streamer:
            response += character
            _chatbot[-1] = (chat_query, response)
            yield _chatbot
            full_response = response

        # for response in model.chat_stream(tokenizer, message, history=history):
        #     _chatbot[-1] = (chat_query, response)
        #
        #     yield _chatbot
        #     full_response = response

        response = full_response
        _chatbot[-1] = (chat_query, response)

        task_history[-1] = (query, full_response)
        print("Qwen-VL-Chat: " + full_response)
        yield _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        if text == "":
            text = random.choice(template)
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(text, None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    def convert(history):
        response = history[-1][1]
        docx = Docx()
        pretty_html = docx.pretty(response, 2, None)
        path = r'/mnt/ceph/Vary/images/test.html'
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
        chatbot = gr.Chatbot(label='', elem_classes="control-height", height=750)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image"])
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            convert_btn = gr.Button("🧹 Convert (转换)")
        file_download = gr.File(label="Download Link")

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)
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
