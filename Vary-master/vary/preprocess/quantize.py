import torch
from transformers import AutoTokenizer
from vary.model import varyQwenForCausalLM, varyConfig


def quantize_awq():
    from awq import AutoAWQForCausalLM
    model_path = '/mnt/ceph2/Vary/runs/0731/checkpoint-55000'
    quant_path = '/mnt/ceph2/Vary/runs/0731/awq'
    # model_path = '/mnt/ceph2/pretrained/llava-hf/llava-1.5-7b-hf'
    # quant_path = '/mnt/ceph2/pretrained/llava-hf/llava-1.5-7b-hf/awq'
    # model_path = '/mnt/ceph2/pretrained/Qwen/Qwen2-7B-Instruct'
    # quant_path = '/mnt/ceph2/pretrained/Qwen/Qwen2-7B-Instruct/awq'
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

    # Load model
    # model = AutoAWQForCausalLM.from_pretrained(
    #     model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
    # )
    # model = AutoAWQForCausalLM.from_pretrained(model_path, device_map={"": "cpu"}, **{"low_cpu_mem_usage": True})
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, device_map="cuda", **{"low_cpu_mem_usage": True}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

    print(f'Model is quantized and saved at "{quant_path}"')


def quantize_gptq():
    from auto_gptq import BaseQuantizeConfig
    from auto_gptq.modeling import BaseGPTQForCausalLM, Qwen2GPTQForCausalLM

    pretrained_model_dir = "/mnt/ceph2/Vary/runs/0711_fp32"
    quantized_model_dir = "/mnt/ceph2/Vary/runs/0711_fp32/gptq-8bit"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = BaseQuantizeConfig(
        bits=8,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
        damp_percent=0.1,
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = Qwen2GPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir)
    tokenizer.save_pretrained(quantized_model_dir)


def bf16_to_fp16():
    # model_name = "/mnt/ceph2/Vary/runs/0711"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # config = varyConfig.from_pretrained(model_name, trust_remote_code=True)
    # config.pad_token_id = tokenizer.eos_token_id
    # from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    # with init_empty_weights():
    #     model = varyQwenForCausalLM._from_config(config, torch_dtype=torch.float32)
    # no_split_modules = model._no_split_modules
    # print(f"no_split_modules: {no_split_modules}", flush=True)
    # map_list = {0: "40GB"}
    # device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)
    # print(device_map)
    # model = load_checkpoint_and_dispatch(model, checkpoint=model_name, device_map=device_map)
    # model.save_pretrained("/mnt/ceph2/Vary/runs/0711_fp32")
    # tokenizer.save_pretrained("/mnt/ceph2/Vary/runs/0711_fp32")

    from safetensors.torch import load_file, save_file
    import numpy as np
    from tqdm import tqdm
    float16_max = np.finfo(np.float16).max
    tensors = load_file("/mnt/ceph2/Vary/runs/0711/gptq-8bit/model.safetensors")
    processed_tensors = {}
    global_max = -float('inf')
    for key, tensor in tqdm(tensors.items()):
        if tensor.dtype == torch.float16:
            tensor_float32 = tensor.to(torch.float32)
            # if (tensor_float32 > float16_max).any():
            #     print(f"Tensor {key} has values exceeding the float16 max value.")
            tensor_max = tensor_float32.max().item()
            if tensor_max > global_max:
                global_max = tensor_max
    print(global_max)

    # for key, tensor in tensors.items():
    #     if tensor.dtype == torch.bfloat16:
    #         tensor = tensor.to(torch.float32).to(torch.float16)
    #
    #     processed_tensors[key] = tensor
    #
    # save_file(processed_tensors, "/mnt/ceph2/Vary/runs/0711/gptq-8bit-2/model.safetensors")


if __name__ == "__main__":
    bf16_to_fp16()
