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
from lxml import html
from PIL import Image
from io import BytesIO
from vary.model.plug.blip_process import BlipImageEvalProcessor
from transformers import TextStreamer
from vary.model.plug.transforms import train_transform, test_transform
from convertor.docx2html import DocxConvertor

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


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained('/mnt/ceph/Vary/runs/0327/', trust_remote_code=True)

    # model = varyQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True).eval()
    config = varyConfig.from_pretrained(model_name, trust_remote_code=True)
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    with init_empty_weights():
        model = varyQwenForCausalLM._from_config(config, torch_dtype=torch.float16)
    no_split_modules = model._no_split_modules
    print(f"no_split_modules: {no_split_modules}", flush=True)
    map_list = {0: "24GB"}
    device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)
    model = load_checkpoint_and_dispatch(model, checkpoint=model_name, device_map=device_map).eval()


    model.to(device='cuda',  dtype=torch.bfloat16)


    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)

    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256

    # qs = 'Provide the ocr results of this image.'
    # qs = 'Convert the document to markdown formart.'
    # qs = 'Convert the document to html formart:'
    # qs = 'Detect the chart:'
    # qs = 'Make the document into an HTML format:'
    # qs = ''
    prompts = ["Provide the ocr results of this image:",
               "Convert the document to html formart:",
               "Convert the document to html/latex formart:"]

    print('stop_str:', tokenizer.convert_tokens_to_ids(['</s>']))
    print(tokenizer('</s>').input_ids)

    # conv_mode = "mpt"
    # args.conv_mode = conv_mode

    convertor = DocxConvertor()
    while True:
        # try:
        image_file = input("Enter image file: ")
        if not image_file:
            break
        # k = int(input("Enter prompt number: "))
        k = 2
        qs = prompts[k]
        # qs = 'Detect and mark the locations of all graphical content in the provided document picture.'
        # k = input("Enter prompt number: ").strip()
        # qs = k
        # qs = 'Find every illustration in this document image, presenting their exact positions in the structured format @[xmin, ymin, xmax, ymax]@.'
        # qs = input("prompt: ")
        print(qs)
        if use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN  + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        print('stop_str:', stop_str)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        if image_file[0] == "'":
            image_file = image_file[1:-2]
        image = load_image(image_file)
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
            print(len(outputs))

            pretty_html = convertor.pretty(outputs, k, image_file)
            with open(r'test.html', 'w', encoding='utf-8') as f:
                f.write(pretty_html)
        # except Exception as e:
        #     print(e)
        #     continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--conv-mode", type=str, default=None)
    args = parser.parse_args()

    eval_model(args)
