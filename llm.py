import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaForSequenceClassification, LlamaConfig, Trainer, TrainingArguments, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, default_data_collator, TrainerCallback
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
import os
from peft import PeftModelForSequenceClassification, get_peft_config
from utils.args import Arguments

from data.load import load_data
from data.dataset import NCDataset
from utils.peft import create_peft_config


def collect_txt(idx, txt):
    tmp = []
    for i in idx:
        tmp.append(txt[i])
    return tmp


def load_llm(name='llama'):
    path = {
        'llama': "meta-llama/Llama-2-7b-hf", # https://huggingface.co/docs/transformers/main/model_doc/llama2
        'baichuan': "baichuan-inc/Baichuan2-7B-Base",
        'vicuna': "lmsys/vicuna-7b-v1.5"
    }[name]
    
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16, num_labels=num_classes)
    
    # model = model.model # only use encoder
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model


if __name__ == '__main__':
    args = Arguments().parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = f"tmp"
    epochs = args.epochs
    enable_profiler = False
    
    acc_list = []
    for i in  range(5):
        data, text, num_classes = load_data(args.dataset, use_text=True, seed=i)
        
        # Load model from HuggingFace Hub
        tokenizer, model = load_llm(name='llama')
        # X = tokenizer(text, padding=True, truncation=True, max_length=512)
        
        # Set up profiler
        if enable_profiler:
            wait, warmup, active, repeat = 1, 1, 2, 1
            total_steps = (wait + warmup + active) * (1 + repeat)
            schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
            profiler = torch.profiler.profile(
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
                record_shapes=True,
                profile_memory=True,
                with_stack=True)
            
            class ProfilerCallback(TrainerCallback):
                def __init__(self, profiler):
                    self.profiler = profiler
                    
                def on_step_end(self, *args, **kwargs):
                    self.profiler.step()

            profiler_callback = ProfilerCallback(profiler)
        else:
            profiler = nullcontext()

        model, lora_config = create_peft_config(model, method=args.peft)

        config = {
            'lora_config': lora_config,
            'learning_rate': args.lr,
            'num_train_epochs': epochs,
            'gradient_accumulation_steps': 2,
            'per_device_train_batch_size': args.batch_size,
            'gradient_checkpointing': False,
        }
        
        train_idx = data.train_mask.nonzero().squeeze().tolist()
        val_idx = data.val_mask.nonzero().squeeze().tolist()
        test_idx = data.test_mask.nonzero().squeeze().tolist()
        train_txt = collect_txt(train_idx, text)
        val_txt = collect_txt(val_idx, text)
        test_txt = collect_txt(test_idx, text)
        
        train_encodings = tokenizer(train_txt, truncation=True, padding=True, return_tensors="pt", max_length=512).to("cuda")
        val_encodings = tokenizer(val_txt, truncation=True, padding=True, return_tensors="pt", max_length=512).to("cuda")
        test_encodings = tokenizer(test_txt, truncation=True, padding=True, return_tensors="pt", max_length=512).to("cuda")

        train_dataset = NCDataset(train_encodings, data.y[train_idx])
        val_dataset = NCDataset(val_encodings, data.y[val_idx])
        test_dataset = NCDataset(test_encodings, data.y[test_idx])
        
        # Training
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            bf16=True,  # Use BF16 if available
            dataloader_pin_memory=False,
            # logging strategies
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="no",
            optim="adamw_torch_fused",
            max_steps=total_steps if enable_profiler else -1,
            **{k:v for k,v in config.items() if k != 'lora_config'}
        )

        with profiler:
            # Create Trainer instance
            # model.score.weight.requires_grad = True # need to train the classifier
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset = val_dataset,
                data_collator=default_data_collator,
                callbacks=[profiler_callback] if enable_profiler else [],
            )

        # Start training
        trainer.train()
            

        predictions = trainer.predict(test_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        acc = (preds == predictions.label_ids).sum()/len(predictions.label_ids)
        print(i,acc)
        acc_list.append(acc)
        del predictions
        del trainer
        del model # clear cache
        del tokenizer
        torch.cuda.empty_cache()
        
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc*100:.2f}±{final_acc_std*100:.2f}")
