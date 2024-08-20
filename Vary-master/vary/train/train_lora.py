import transformers
from awq import AutoAWQForCausalLM
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    CLIPImageProcessor
)
from peft import get_peft_model, LoraConfig, TaskType
from vary.data import make_supervised_data_module
from vary.utils.arguments import *
from vary.model.plug.transforms import train_transform2


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model
    model = AutoAWQForCausalLM.from_quantized(model_args.model_name_or_path, fuse_layers=False, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # tokenizer.pad_token = tokenizer.eos_token
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, padding_side="right",
                                              model_max_length=training_args.model_max_length, )

    data_args.image_token_len = 256
    data_args.image_processor = CLIPImageProcessor.from_pretrained('/cache/vit-large-patch14/')
    data_args.image_processor_high = train_transform2
    data_args.use_im_start_end = model_args.use_im_start_end

    # Prepare data
    data_module = make_supervised_data_module(
            interleave=training_args.interleave,
            with_box=training_args.with_box,
            tokenizer=tokenizer,
            data_args=data_args
        )

    # Config Lora
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.5,
        target_modules=["q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        "lm_head",],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )

    model = get_peft_model(model.model, lora_config)

    model.print_trainable_parameters()

    # training_arguments = TrainingArguments(
    #     output_dir="./output",
    #     per_device_train_batch_size=1,
    #     optim="adamw_torch",
    #     num_train_epochs=1,
    #     learning_rate=1e-4,
    #     evaluation_strategy="no",
    #     save_strategy="epoch",
    #     save_steps=100,
    #     logging_steps=50,
    #     eval_steps=None,
    #     load_best_model_at_end=False
    # )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
