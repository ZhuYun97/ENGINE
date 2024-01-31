from transformers import AutoTokenizer, AutoModel, Trainer, default_data_collator, TrainerCallback, TrainingArguments
from contextlib import nullcontext
import torch
import os
import numpy as np
from utils.args import Arguments
from models.LMs import BertClassifier
from data.load import load_data
from data.dataset import NCDataset


def collect_txt(idx, txt):
    tmp = []
    for i in idx:
        tmp.append(txt[i])
    return tmp

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':
    config = Arguments().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_id = {
        'sentencebert': 'sentence-transformers/bert-base-nli-mean-tokens',
        'deberta': "microsoft/deberta-base",
        'bert': 'bert-base-uncased'
    }
    
    output_dir = f"tmp"
    epochs = config.epochs # 4
    enable_profiler = False
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
    
    acc_list = []
    for i in  range(5):
        data, text, num_classes = load_data(config.dataset, use_text=True, seed=i)
        
        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(mode_id[config.lm_type])
        bert_model = AutoModel.from_pretrained(mode_id[config.lm_type], output_hidden_states=True, return_dict=True)

        model = BertClassifier(bert_model, num_classes)
        # X = tokenizer(text, padding=True, truncation=True, max_length=512)
        
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
        
        # Define training args
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            bf16=False,  # Use BF16 if available
            dataloader_pin_memory=False,
            # logging strategies
            logging_dir=f"{output_dir}/logs",
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="no",
            optim="adamw_torch_fused",
            max_steps=total_steps if enable_profiler else -1,
            learning_rate=config.lr, # 5e-5
            num_train_epochs=epochs,
            gradient_accumulation_steps=2,
            per_device_train_batch_size=config.batch_size, # 8, 6for deberta
            gradient_checkpointing=False,
            local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        )

        with profiler:
            # Create Trainer instance
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
        print(i, acc)
        acc_list.append(acc)
        # model.save_pretrained(output_dir)
        
    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    print(f"# final_acc: {final_acc*100:.2f}±{final_acc_std*100:.2f}")
    