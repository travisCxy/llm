import torch
import torch.nn as nn
from PIL import Image
from transformers import Qwen2ForCausalLM, Qwen2Config, AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, TextStreamer, CLIPImageProcessor
from vary.model import varyQwenForCausalLM, varyConfig
from vary.model.vision_encoder.sam import build_sam_vit_l
from vary.model.plug.blip_process import BlipImageEvalProcessor

multimodal_model_path = "/mnt/ceph2/Vary/runs/0711"
language_model_path = "/mnt/ceph2/Vary/runs/0711/split/lm_model"
vision_model_path = "/mnt/ceph2/Vary/runs/0711/split/vision_model.pth"

def split():
    # # 加载训练后的多模态模型
    multimodal_model = varyQwenForCausalLM.from_pretrained(multimodal_model_path)
    tokenizer = AutoTokenizer.from_pretrained(multimodal_model_path)

    # 保存tokenizer
    tokenizer.save_pretrained(language_model_path)

    # 提取语言模型部分
    lm_config = Qwen2Config.from_dict(multimodal_model.config.to_dict())
    lm_model = Qwen2ForCausalLM(lm_config)
    lm_model.load_state_dict(multimodal_model.state_dict(), strict=False)

    # # 确保lm_head被正确加载
    # lm_model.lm_head = multimodal_model.lm_head

    # 保存语言模型
    lm_model.to('cpu', torch.bfloat16)
    lm_model.save_pretrained(language_model_path)

    # 提取视觉模型部分
    vision_tower = multimodal_model.model.vision_tower
    vision_tower_high = multimodal_model.model.vision_tower_high
    mm_projector = multimodal_model.model.mm_projector
    mm_projector_vary = multimodal_model.model.mm_projector_vary

    # 保存视觉模型及多模态投影器
    torch.save({
        'vision_tower_state_dict': vision_tower.state_dict(),
        'vision_tower_high_state_dict': vision_tower_high.state_dict(),
        'mm_projector_state_dict': mm_projector.state_dict(),
        'mm_projector_vary_state_dict': mm_projector_vary.state_dict(),
    }, vision_model_path)

    print("Model splitting and saving completed.")


def get_prompt():
    from vary.utils.conversation import conv_templates
    with open('ocr.txt', 'r', encoding='utf-8') as f:
        ocr = f.read()
    qs = '<img>' + '<imgpad>' * 256 + '</img>' + f'\nOCR:\n{ocr}\nConvert options: [font, id].\nConvert:'
    conv = conv_templates["mpt"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt


def test():
    # 1. 加载语言模型
    config = varyConfig.from_pretrained(language_model_path)
    tokenizer = AutoTokenizer.from_pretrained(language_model_path)
    lm_model = varyQwenForCausalLM.from_pretrained(language_model_path, config=config)

    # 2. 加载视觉模型和多模态投影器
    vision_state_dict = torch.load(vision_model_path)

    vision_tower_state_dict = vision_state_dict['vision_tower_state_dict']
    vision_tower_high_state_dict = vision_state_dict['vision_tower_high_state_dict']
    mm_projector_state_dict = vision_state_dict['mm_projector_state_dict']
    mm_projector_vary_state_dict = vision_state_dict['mm_projector_vary_state_dict']

    # 3. 将视觉模型和投影器的权重分配给多模态模型
    lm_model.model.vision_tower.load_state_dict(vision_tower_state_dict)
    lm_model.model.vision_tower_high.load_state_dict(vision_tower_high_state_dict)
    lm_model.model.mm_projector.load_state_dict(mm_projector_state_dict)
    lm_model.model.mm_projector_vary.load_state_dict(mm_projector_vary_state_dict)

    # from tqdm import tqdm
    # for key in tqdm(multimodal_model.state_dict().keys()):
    #     # if 'mm_projector' not in key:
    #     #     continue
    #     if not torch.equal(multimodal_model.state_dict()[key], lm_model.state_dict()[key]):
    #         print(key)

    lm_model.to(device='cuda',  dtype=torch.bfloat16)

    print("Multimodal model reloaded successfully.")

    streamer = TextStreamer(tokenizer, skip_prompt=True)
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    prompt = get_prompt()
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    image = Image.open("/mnt/ceph2/Vary/转word测试数据/数学/2年级-317225836-whiten.jpg").convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    image_tensor_1 = image_processor_high(image)
    images = [(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())]

    with torch.autocast("cuda", dtype=torch.bfloat16):
        generation_output = lm_model.generate(
            input_ids=input_ids,
            images=images,
            max_new_tokens=2048,
            streamer=streamer
        )


def test2():
    lm_model = Qwen2ForCausalLM.from_pretrained(language_model_path)
    vision_tower = CLIPVisionModel.from_pretrained('/cache/vit-large-patch14/')
    vision_tower_high = build_sam_vit_l()
    mm_projector = nn.Linear(1024, 1792)
    mm_projector_vary = nn.Linear(1024, 1792)

    vision_state_dict = torch.load(vision_model_path)
    vision_tower_state_dict = vision_state_dict['vision_tower_state_dict']
    vision_tower_high_state_dict = vision_state_dict['vision_tower_high_state_dict']
    mm_projector_state_dict = vision_state_dict['mm_projector_state_dict']
    mm_projector_vary_state_dict = vision_state_dict['mm_projector_vary_state_dict']
    vision_tower.load_state_dict(vision_tower_state_dict)
    vision_tower_high.load_state_dict(vision_tower_high_state_dict)
    mm_projector.load_state_dict(mm_projector_state_dict)
    mm_projector_vary.load_state_dict(mm_projector_vary_state_dict)

    lm_model.to(device='cuda',  dtype=torch.bfloat16)
    vision_tower.to(device='cuda',  dtype=torch.bfloat16)
    vision_tower_high.to(device='cuda',  dtype=torch.bfloat16)
    mm_projector.to(device='cuda',  dtype=torch.bfloat16)
    mm_projector_vary.to(device='cuda',  dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(language_model_path)
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)

    prompt = get_prompt()
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    image = Image.open("/mnt/ceph2/Vary/转word测试数据/数学/2年级-317225836-whiten.jpg").convert('RGB')
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    image_tensor_1 = image_processor_high(image)
    images = [(image_tensor.unsqueeze(0).to(torch.bfloat16).cuda(), image_tensor_1.unsqueeze(0).to(torch.bfloat16).cuda())]

    vision_select_layer = -2
    image_features_1 = []
    image_features_2 = []
    for image in images:
        image_forward_out = vision_tower(image[0], output_hidden_states=True)
        select_hidden_state = image_forward_out.hidden_states[vision_select_layer]
        image_feature = select_hidden_state[:, 1:]

        cnn_feature = vision_tower_high(image[1])
        cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1)

        image_features_1.append(image_feature)
        image_features_2.append(cnn_feature)

    image_features_1 = [mm_projector(image_feature) for image_feature in image_features_1]
    image_features_2 = [mm_projector_vary(image_feature) for image_feature in image_features_2]
    image_features = [torch.cat((image_feature[0], image_feature[1]), dim=-1) for image_feature in zip(image_features_1, image_features_2)]

    im_start_token = 151647
    im_end_token = 151648
    inputs_embeds = lm_model.get_input_embeddings()(input_ids)
    new_input_embeds = []
    for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
        if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
            raise ValueError("The number of image start tokens and image end tokens should be the same.")
        image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
        for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
            num_patches = per_cur_image_features.shape[0]
            print(image_start_token_pos, num_patches)
            if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                raise ValueError("The image end token should follow the image start token.")
            cur_input_embeds = torch.cat(
                (
                    cur_input_embeds[:image_start_token_pos + 1],
                    per_cur_image_features,
                    cur_input_embeds[image_start_token_pos + num_patches + 1:]
                ),
                dim=0
            )
        new_input_embeds.append(cur_input_embeds)
    inputs_embeds = torch.stack(new_input_embeds, dim=0)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        generation_output = lm_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=2048,
            streamer=streamer
        )


if __name__ == '__main__':
    # test2()
    ids = [151644, 8948, 198, 2610, 1265, 1795, 279, 11221, 15516, 323, 10339, 697, 11253, 304, 7716, 13, 151645, 151644, 872, 198, 151647, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151646, 151648, 198, 27, 1097, 877, 428, 21, 755, 40820, 107718, 104552, 17447, 106652, 111593, 107090, 100199, 44991, 522, 1097, 1784, 2011, 2235, 18, 23, 15, 11, 23, 16, 23547, 22, 19, 20, 11, 16, 15, 17, 12533, 2011, 397, 27, 1097, 877, 428, 16, 16, 755, 107278, 22035, 522, 1097, 1784, 2011, 2235, 18, 17, 24, 11, 16, 16, 22, 23547, 20, 17, 21, 11, 16, 19, 15, 12533, 2011, 397, 27, 1097, 877, 428, 16, 17, 755, 66187, 25, 716, 522, 1097, 1784, 2011, 2235, 20, 24, 23, 11, 16, 16, 22, 23547, 22, 24, 17, 11, 16, 18, 24, 12533, 2011, 397, 27, 1097, 877, 428, 16, 755, 16, 13, 102208, 90840, 115865, 16, 17, 119120, 11, 56006, 110029, 82647, 90840, 21, 119120, 11, 90840, 110029, 100430, 119120, 30, 90840, 9370, 115439, 8863, 56006, 110029, 42140, 16, 19, 522, 1097, 1784, 2011, 2235, 16, 16, 23, 11, 16, 20, 18, 23547, 24, 24, 15, 11, 16, 22, 20, 12533, 2011, 397, 27, 1097, 877, 428, 22, 755, 119120, 11, 90840, 115439, 100430, 119120, 26055, 1097, 1784, 2011, 2235, 16, 16, 24, 11, 16, 22, 24, 23547, 18, 19, 16, 11, 17, 15, 15, 12533, 2011, 397, 27, 1097, 877, 428, 18, 755, 17, 13, 98641, 90840, 102163, 112398, 11, 100043, 39953, 98641, 34187, 20, 20, 100509, 11, 100916, 39953, 56006, 100043, 39953, 42140, 98641, 21, 100509, 11, 106980, 39953, 56006, 100916, 39953, 82647, 98641, 18, 100509, 90088, 1097, 1784, 2011, 2235, 16, 17, 19, 11, 17, 24, 24, 23547, 24, 24, 21, 11, 18, 16, 24, 12533, 2011, 397, 27, 1097, 877, 428, 23, 755, 106980, 39953, 98641, 110599, 100509, 26055, 1097, 1784, 2011, 2235, 16, 17, 22, 11, 18, 17, 17, 23547, 18, 16, 23, 11, 18, 19, 18, 12533, 2011, 397, 27, 1097, 877, 428, 15, 755, 18, 13, 106899, 18830, 99737, 100041, 16, 20, 102307, 11, 56006, 100213, 100088, 100041, 82647, 22, 102307, 11, 100213, 100088, 100041, 56006, 108633, 100041, 82647, 19, 102307, 11, 108633, 100041, 18830, 99195, 102307, 26055, 1097, 1784, 2011, 2235, 16, 17, 22, 11, 19, 18, 20, 23547, 24, 19, 22, 11, 19, 21, 16, 12533, 2011, 397, 27, 1097, 877, 428, 17, 755, 19, 13, 18830, 19, 15, 99922, 110548, 11, 17177, 89012, 23, 18947, 106798, 1773, 108156, 17177, 18, 99922, 11, 97706, 100124, 100430, 99922, 26055, 1097, 1784, 2011, 2235, 16, 18, 17, 11, 20, 19, 22, 23547, 23, 17, 17, 11, 20, 22, 19, 12533, 2011, 397, 27, 1097, 877, 428, 19, 755, 20, 13, 107500, 17447, 101221, 18830, 19, 17, 17340, 11, 26939, 109779, 102077, 13343, 11, 109217, 9370, 56006, 17447, 39953, 9370, 42140, 19, 17340, 11, 99601, 39953, 106500, 522, 1097, 1784, 2011, 2235, 16, 19, 18, 11, 21, 21, 15, 23547, 24, 22, 18, 11, 21, 23, 15, 12533, 2011, 397, 27, 1097, 877, 428, 16, 15, 755, 99195, 17340, 26055, 1097, 1784, 2011, 2235, 16, 19, 19, 11, 21, 23, 16, 23547, 17, 15, 23, 11, 22, 15, 15, 12533, 2011, 397, 27, 1097, 877, 428, 20, 755, 21, 13, 105670, 100343, 101221, 27366, 18830, 105982, 17, 20, 17340, 11, 26939, 70790, 33447, 109217, 16, 17, 17340, 11, 17447, 39953, 23, 17340, 1773, 99601, 108220, 105982, 522, 1097, 1784, 2011, 2235, 16, 19, 24, 11, 22, 23, 22, 23547, 24, 21, 20, 11, 23, 15, 21, 12533, 2011, 397, 27, 1097, 877, 428, 24, 755, 56006, 101221, 82647, 100430, 17340, 26055, 1097, 1784, 2011, 2235, 16, 20, 15, 11, 23, 15, 23, 23547, 18, 18, 16, 11, 23, 17, 23, 12533, 2011, 1339, 12012, 2606, 25, 508, 4026, 11, 877, 26126, 12012, 25, 151645, 151644, 77091, 198]
    tokenizer = AutoTokenizer.from_pretrained(language_model_path)
    print(tokenizer.decode(ids))
