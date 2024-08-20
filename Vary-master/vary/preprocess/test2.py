import torch
from transformers import AutoTokenizer
from vary.model.vary_qwen2_vary import varyQwenForCausalLM, varyConfig


if __name__ == "__main__":
    model_name = '0711'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    config = varyConfig.from_pretrained(model_name, trust_remote_code=True)
    config.pad_token_id = tokenizer.eos_token_id
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
    with init_empty_weights():
        model = varyQwenForCausalLM._from_config(config, torch_dtype=torch.float16)
    no_split_modules = model._no_split_modules
    print(f"no_split_modules: {no_split_modules}", flush=True)
    map_list = {0: "11GB", 1: "11GB"}
    device_map = infer_auto_device_map(model, max_memory=map_list, no_split_module_classes=no_split_modules)
    print(device_map)
    model = load_checkpoint_and_dispatch(model, checkpoint=model_name, device_map=device_map).eval()

    model.to(device='cuda',  dtype=torch.bfloat16)
