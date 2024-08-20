import asyncio
import os.path
import time
from PIL import Image
from vary.utils.conversation import conv_templates
from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.lora.request import LoRARequest
from vllm_qwen2 import VaryQwen2ForConditionalGeneration
ModelRegistry.register_model("varyQwenForCausalLM", VaryQwen2ForConditionalGeneration)


def get_prompt(k):
    # token: 880+460 839+323 2017+970 3298+799 548+389
    files = ['2年级-317225836-whiten', '4年级-313999843-whiten', '初一-313062998-whiten',
             '3年级-317768441-whiten', '1年级-337983156-whiten']
    with open(f'/mnt/ceph2/Vary/转word测试数据/数学/{files[k]}-0711/ocr.txt', 'r', encoding='utf-8') as f:
        ocr = f.read()
    qs = f'<imgpad>\n{ocr}\nConvert options: [font, id].\nConvert:'
    # qs = '<img>' + '<imgpad>' * 256 + '</img>' + f'\nOCR:\n{ocr}\nConvert options: [font, id].\nConvert:'
    # qs = f'<imgpad>'
    conv = conv_templates["mpt"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image = Image.open(f"/mnt/ceph2/Vary/转word测试数据/数学/{files[k]}.jpg")
    prompt = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            }
    return prompt


def main(stream):
    sampling_params = SamplingParams(
        temperature=0,
        stop=['<|im_end|>'],
        max_tokens=2048,
    )
    model = "/mnt/ceph2/Vary/runs/0711/gptq-8bit"
    quantization = "gptq"
    dtype = 'bfloat16'
    max_num_seqs = 1
    max_model_len = 4096
    gpu_memory_utilization = 0.5

    if stream:
        import uuid
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        engine_args = AsyncEngineArgs(
            model=model,
            tensor_parallel_size=2,
            quantization=quantization,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            # speculative_model="[ngram]",
            # num_speculative_tokens=5,
            # ngram_prompt_lookup_max=4,
            # use_v2_block_manager=True,
            # enforce_eager=True,
            # disable_log_requests=True,
            # disable_log_stats=True,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        start_time = time.time()
        # request_id = uuid.uuid4().hex
        # results_generator = engine.generate(prompt, sampling_params, request_id)
        #
        # async def stream_results():
        #     last_index = 0
        #     async for request_output in results_generator:
        #         text = request_output.outputs[0].text
        #         print(text[last_index:], end="", flush=True)
        #         last_index = len(text)
        # asyncio.run(stream_results())

        async def generate(prompt):
            request_id = uuid.uuid4().hex
            results_generator = engine.generate(prompt, sampling_params, request_id)
            # results_generator = iterate_with_cancellation(results_generator, is_cancelled=request.is_disconnected)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            return request_id, final_output

        async def process(prompts):
            tasks = [generate(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            for request_id, result in results:
                print(request_id, result.prompt)
                print(result.outputs[0].text)
                print('prompt token:', len(result.prompt_token_ids))
                print('generate token:', len(result.outputs[0].token_ids))

        prompts = [get_prompt(k) for k in range(5)]
        asyncio.run(process(prompts))
        print(f"Time taken: {time.time() - start_time:.2f}s")
    else:
        llm = LLM(model=model, dtype=dtype,
                  enforce_eager=True,
                  cpu_offload_gb=15,
                  # enable_lora=True,
                  tensor_parallel_size=1,
                  max_num_seqs=max_num_seqs, max_model_len=max_model_len,
                  gpu_memory_utilization=gpu_memory_utilization,
                  # speculative_model="[ngram]",
                  # num_speculative_tokens=5,
                  # ngram_prompt_lookup_max=4,
                  # use_v2_block_manager=True,
                  )
        start_time = time.time()
        for k in range(5):
            prompt = get_prompt(k)
            outputs = llm.generate(prompt, sampling_params=sampling_params,
                                   # lora_request=LoRARequest("sql_adapter", 1, '/mnt/ceph2/Vary/runs/0805')
                                   )
            for o in outputs:
                print(o.outputs[0].text)
                print('prompt token:', len(o.prompt_token_ids))
                print('generate token:', len(o.outputs[0].token_ids))
        print(f"Time taken: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main(False)

    # import torch
    # import requests
    # from PIL import Image
    # from awq import AutoAWQForCausalLM
    # from transformers import AutoTokenizer, TextStreamer, CLIPImageProcessor, AutoProcessor
    # from vary.model.plug.blip_process import BlipImageEvalProcessor

    # quant_path = "/mnt/ceph2/pretrained/Qwen/Qwen2-7B-Instruct/awq"
    # model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=True)
    # tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # prompt = "You're standing on the surface of the Earth. " \
    #          "You walk one mile south, one mile west and one mile north. " \
    #          "You end up exactly where you started. Where are you?"
    # tokens = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    # generation_output = model.generate(
    #     tokens,
    #     streamer=streamer,
    #     max_new_tokens=512
    # )

    # quant_path = '/mnt/ceph2/pretrained/llava-hf/llava-1.5-7b-hf/awq'
    # model = AutoAWQForCausalLM.from_quantized(quant_path)
    # processor = AutoProcessor.from_pretrained(quant_path)
    # streamer = TextStreamer(processor, skip_prompt=True)
    #
    # # Define prompt
    # prompt = """\
    # <|im_start|>system\nAnswer the questions.<|im_end|>
    # <|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|>
    # <|im_start|>assistant
    # """
    #
    # # Define image
    # image = Image.open("/mnt/ceph2/apple.jpeg")
    #
    # # Load inputs
    # inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)
    # generation_output = model.generate(
    #     **inputs,
    #     max_new_tokens=512,
    #     streamer=streamer
    # )

    # quant_path = "/mnt/ceph2/Vary/runs/0711/awq"
    # model = AutoAWQForCausalLM.from_quantized(quant_path, fuse_layers=False)
    # tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
    # streamer = TextStreamer(tokenizer, skip_prompt=True)
    # image_processor = CLIPImageProcessor.from_pretrained("/cache/vit-large-patch14/", torch_dtype=torch.float16)
    # image_processor_high = BlipImageEvalProcessor(image_size=1024)
    #
    # prompt = get_prompt()
    # inputs = tokenizer([prompt])
    # input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # image = Image.open("/mnt/ceph2/Vary/转word测试数据/数学/2年级-317225836-whiten.jpg").convert('RGB')
    # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    # image_tensor_1 = image_processor_high(image)
    # images = [(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())]
    #
    # start_time = time.time()
    # generation_output = model.generate(
    #     input_ids=input_ids,
    #     images=images,
    #     max_new_tokens=2048,
    #     streamer=streamer
    # )
    # print(tokenizer.batch_decode(generation_output, skip_special_tokens=True))
    # print(f"Time taken: {time.time() - start_time:.2f}s")
    # print(f"Tokens per second: {(generation_output.shape[1] - input_ids.shape[1]) / (time.time() - start_time):.2f}")

    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # device = "cuda"  # the device to load the model onto
    # model_name = "/mnt/ceph2/pretrained/Qwen/Qwen1.5-0.5B-Chat"
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prompt = "Give me a short introduction to large language model."
    # messages = [{"role": "user", "content": prompt}]
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False, prompt_lookup_num_tokens=10)
    # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)
