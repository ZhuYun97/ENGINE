
from data.load import load_data
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaModel, LlamaConfig, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import torch
import math

from utils.args import Arguments


def collect_txt(idx, txt):
    tmp = []
    for i in idx:
        tmp.append(txt[i])
    return tmp

def process_text(text):
    refined_txt_arr = []
    for txt in text:
        refined_txt = txt.split('\n')[1]
        assert refined_txt[:10] == 'Abstract: '
        refined_txt_arr.append(refined_txt[10:])
    return refined_txt_arr


def save_hidden_states(path, max_length=512, llm_model='llama'):
    assert llm_model in ['llama', 'bert', 'baichuan', 'vicuna']

    if llm_model == 'baichuan':
        tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Base", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Base", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, output_hidden_states=True, return_dict=True)
        model = model.model # only use encoder
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        hidden_layers = len(model.layers)
    elif llm_model == 'vicuna':
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, output_hidden_states=True, return_dict=True)
        model = model.model # only use encoder
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        hidden_layers = len(model.layers)

    elif llm_model == 'llama':
        # Llama
        token_id="meta-llama/Llama-2-7b-hf"
        model_id="meta-llama/Llama-2-7b-hf"

        tokenizer = LlamaTokenizer.from_pretrained(token_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = LlamaModel.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16, output_hidden_states=True, return_dict=True)
        model.config.pad_token_id = model.config.eos_token_id
        
        hidden_layers = len(model.layers)
    # ***************************************************************

    elif llm_model == 'bert':
        # Sentence BERT
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens', output_hidden_states=True, return_dict=True).cuda()
        hidden_layers = 12
    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # **************************************************************

    batch_size = 8
    model.eval()
    layers = [[] for i in range(hidden_layers+1)]
    for i in tqdm(range(math.ceil(len(text)/batch_size))):
        if (i+1)*batch_size <= len(text):
            txt = text[(i)*batch_size: (i+1)*batch_size]
        else:
            txt = text[(i)*batch_size:]
        # txt = process_text(txt)
        
        model_input = tokenizer(txt, truncation=True, padding=True, return_tensors="pt", max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**model_input)
        batch_size = model_input['input_ids'].shape[0]
        sequence_lengths = (torch.eq(model_input['input_ids'], model.config.pad_token_id).long().argmax(-1) - 1)
        hidden_states = out['hidden_states']
        
        
        for i, layer_hid in enumerate(hidden_states):
            layer_hid = layer_hid.cpu()
            # layer_node_hid = layer_hid[torch.arange(batch_size).cpu(), sequence_lengths.cpu()]
            
            # if llm_model == 'bert':
            #     layer_node_hid = layer_hid.permute(1, 0, 2)[0]
            #     layers[i].append(layer_node_hid.cpu())
            # else:
            layer_node_hid = mean_pooling(layer_hid, model_input['attention_mask'].cpu())
            layers[i].append(layer_node_hid.cpu())
            
                
        

    # layers_hid = [torch.stack(xs).float() for xs in layers]
    layers_hid = [torch.cat(xs).float() for xs in layers]

    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(layers_hid, f=os.path.join(path, 'layer_attr.pt'))

if __name__ == '__main__':
    args = Arguments().parse_args()
    
    data, text, num_classes = load_data(args.dataset, use_text=True, use_gpt=False)
    path = f'./llm_cache/{args.dataset}/layers'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_hidden_states(path, 512, 'llama')
