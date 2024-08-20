import argparse
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
import json
from lxml import html
from PIL import Image, ImageDraw
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer
from vary.model.plug.transforms import train_transform, test_transform
from glob import glob
from convertor.docx2html import DocxConvertor
from convertor.html2docx import HtmlConvertor
from convertor.common import get_regions, sort_regions

import asyncio
import time
from PIL import Image
from vary.utils.conversation import conv_templates
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm_qwen2 import VaryQwen2ForConditionalGeneration
ModelRegistry.register_model("varyQwenForCausalLM", VaryQwen2ForConditionalGeneration)

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


def get_ocr(data, image):
    w, h = image.size
    regions, pics, tables = get_regions(data, w, h, merge=True)
    # ocr = ''
    # for region in regions:
    #     ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
    for pic in pics:
        pic['result'] = f'<pic id="{pic["id"]}"/>'
    regions.extend(pics)
    regions.extend(tables)
    regions = sort_regions(regions)
    ocr = ''
    # draw = ImageDraw.Draw(image)
    for region in regions:
        if 'id' in region:
            ocr += f'<ref id="{region["id"]}">{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
        else:
            ocr += f'<ref>{region["result"]}</ref><box>({region["bbox"][0]},{region["bbox"][1]}),({region["bbox"][2]},{region["bbox"][3]})</box>\n'
        # if '<pic' in region['result']:
        #     x1 = region['bbox'][0] * w // 1000
        #     y1 = region['bbox'][1] * h // 1000
        #     x2 = region['bbox'][2] * w // 1000
        #     y2 = region['bbox'][3] * h // 1000
        #     draw.rectangle((x1, y1, x2, y2), fill='gray')
    return ocr


def detect(path, image, output_dir):
    url = "http://10.33.10.63:5003/image_to_json"
    payload = {}
    files = [('image', ('tmp.jpg', open(path, 'rb'), 'image/jpeg'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    data = json.loads(response.text)
    json.dump(data, open(os.path.join(output_dir, 'tmp.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    # data = json.load(open('tmp.json', encoding='utf-8'))
    ocr = get_ocr(data, image)
    # print(ocr)
    with open(os.path.join(output_dir, 'ocr.txt'), 'w', encoding='utf-8') as f:
        f.write(ocr)
    return ocr


def eval_model(args):
    model_name = os.path.expanduser(args.model_name)
    dtype = 'float16'
    max_num_seqs = 1
    max_model_len = 4096
    gpu_memory_utilization = 0.9
    llm = LLM(model=model_name, dtype=dtype,
              max_num_seqs=max_num_seqs, max_model_len=max_model_len,
              gpu_memory_utilization=gpu_memory_utilization,
              speculative_model="[ngram]",
              num_speculative_tokens=5,
              ngram_prompt_lookup_max=4,
              use_v2_block_manager=True,
              )
    sampling_params = SamplingParams(
        temperature=0,
        stop=['<|im_end|>'],
        max_tokens=2048,
    )
    suffix = args.suffix
    with open(f'log{suffix}.txt', 'w', encoding='utf-8') as f:
        for path in glob(args.folder + '**/*.jpg', recursive=True):
            try:
                print(path)
                # if os.path.exists(path[:-4]+'.html'):
                #     continue
                output_dir = os.path.splitext(path)[0] + suffix
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                image = load_image(path)
                ocr = detect(path, image, output_dir)
                qs = f'<imgpad>\n{ocr}\nConvert options: [font, id].\nConvert:'
                conv = conv_templates["mpt"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                image = Image.open(path)
                prompt = {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image
                    },
                }
                start_time = time.time()
                outputs = llm.generate(prompt, sampling_params=sampling_params)[0].outputs[0].text
                cost_time = time.time() - start_time
                with open(os.path.join(output_dir, 'output.txt'), 'w', encoding='utf-8') as f2:
                    f2.write(outputs)
                # convertor = HtmlConvertor(output_dir)
                # convertor.convert(outputs, path[:-4]+f'{suffix}.docx', os.path.join(output_dir, 'ocr.txt'))
                f.write(path + ' ' + str(cost_time) + '\n')
            except Exception as e:
                print(e)
                f.write(str(e) + '\n')
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/mnt/ceph2/Vary/runs/0711_fp32/gptq-8bit")
    parser.add_argument("--folder", type=str, default="/mnt/ceph2/Vary/转word测试数据/数学2/")
    parser.add_argument("--suffix", type=str, default="-0711")
    args = parser.parse_args()

    eval_model(args)
