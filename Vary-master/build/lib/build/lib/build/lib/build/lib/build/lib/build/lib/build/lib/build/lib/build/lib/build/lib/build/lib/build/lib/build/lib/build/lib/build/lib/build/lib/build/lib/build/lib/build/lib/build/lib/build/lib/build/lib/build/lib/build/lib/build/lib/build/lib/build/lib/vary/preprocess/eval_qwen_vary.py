import argparse
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
import json
from lxml import html
from PIL import Image
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer
from vary.model.plug.transforms import train_transform, test_transform
from glob import glob
from convertor.docx2html import DocxConvertor
from convertor.html2docx import HtmlConvertor
from convertor.common import get_regions

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_ocr(data, w, h):
    regions, pics, tables = get_regions(data, w, h, merge=True)
    ocr = ''
    for region in regions:
        ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
    return ocr


def detect(path, w, h):
    url = "http://10.33.10.63:5003/image_to_json"
    payload = {}
    files = [('image', ('tmp.jpg', open(path, 'rb'), 'image/jpeg'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    data = json.loads(response.text)
    json.dump(data, open('output/tmp.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    # data = json.load(open('tmp.json', encoding='utf-8'))
    ocr = get_ocr(data, w, h)
    # print(ocr)
    with open('output/ocr.txt', 'w', encoding='utf-8') as f:
        f.write(ocr)
    return ocr


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # model = varyQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True).eval()
    config = varyConfig.from_pretrained(model_name, trust_remote_code=True)
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    with init_empty_weights():
        model = varyQwenForCausalLM._from_config(config, torch_dtype=torch.float16)
    no_split_modules = model._no_split_modules
    print(f"no_split_modules: {no_split_modules}", flush=True)
    map_list = {1: "24GB"}
    device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)
    model = load_checkpoint_and_dispatch(model, checkpoint=model_name, device_map=device_map).eval()


    model.to(device='cuda',  dtype=torch.bfloat16)


    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)

    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    image_token_len = 256

    conv_mode = "mpt"
    args.conv_mode = conv_mode
    # qs = "Convert the document to html/latex formart:"
    # qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN  + '\n' + qs

    docx_convertor = DocxConvertor()
    html_convertor = HtmlConvertor()
    with open('log1.txt', 'r', encoding='utf-8') as f:
        files = []
        path = f.readline().strip()
        for line in f.readlines():
            if 'list index' in line:
                files.append(path)
            else:
                path = line.strip()
    # files = ['/mnt/ceph2/Vary/转word测试数据/语文/初一-313783975-whiten.jpg']
    with open('log.txt', 'w', encoding='utf-8') as f:
        # for path in glob(args.folder + '**/*-whiten.jpg', recursive=True):
        for path in files:
            try:
                f.write(path + '\n')
                print(path)
                # if os.path.exists(path[:-4]+'.html'):
                #     continue

                image = load_image(path)
                ocr = detect(path, image.size[0], image.size[1])
                # ocr = None
                # print(ocr)
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + f'\nOCR:{ocr}\nConvert with font:'

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt])
                input_ids = torch.as_tensor(inputs.input_ids).cuda()

                # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
                streamer = None

                image_1 = image.copy()
                image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                image_tensor_1 = image_processor_high(image_1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        input_ids,
                        images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                        do_sample=False,
                        num_beams = 1,
                        # temperature=0.2,
                        streamer=streamer,
                        max_new_tokens=2048,
                        stopping_criteria=[stopping_criteria]
                        )
                    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                    # conv.messages[-1][-1] = outputs
                    if outputs.endswith(stop_str):
                        outputs = outputs[:-len(stop_str)]
                    outputs = outputs.strip()
                    # print(len(outputs))

                    pretty_html = docx_convertor.pretty(outputs, 2, path)
                    with open(path[:-4]+'-2.html', 'w', encoding='utf-8') as f2:
                        f2.write(pretty_html)

                    # with open('output/output.txt', 'w', encoding='utf-8') as f2:
                    #     f2.write(outputs)
                    html = outputs.replace('\n', '')
                    convertor = HtmlConvertor()
                    convertor.convert(html, path[:-4]+'-2.docx', 'output/ocr.txt')
            except Exception as e:
                print(e)
                f.write(str(e) + '\n')
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/mnt/ceph2/Vary/runs/0514/")
    parser.add_argument("--conv-mode", type=str, default="mpt")
    parser.add_argument("--folder", type=str, default="/mnt/ceph2/Vary/转word测试数据/")
    args = parser.parse_args()

    eval_model(args)
