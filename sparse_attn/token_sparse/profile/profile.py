import os
import random
import numpy as np
import torch
import math
import argparse
from tqdm import tqdm
import torch
from accelerate import infer_auto_device_map, dispatch_model

from utils import utils
from utils.utils import generate_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.backends.cudnn.benchmark = True
            
def benchmark(model, tokenizer, args):
    # Input Sequence
    context_length_list = [131072]
    delta_dist_list = torch.zeros((len(model.model.layers)))
    for context_length in context_length_list:
        context = generate_prompt(tokenizer, target_len=context_length)
        prompt = tokenizer(context, return_tensors="pt")
        input_ids = prompt['input_ids'].to(model.device)
        attn_mask = prompt['attention_mask'].to(model.device)
        
        # Run
        with torch.no_grad():
            outputs = model(input_ids, attn_mask)
        del outputs
        utils.cleanup_memory()
        
        delta_dist = []
        for i in range(len(model.model.layers)):
            delta_dist.append(model.model.layers[i].delta_dist.mean().detach().cpu().item())
        delta_dist_list += torch.tensor(delta_dist)

    delta_dist_norm = delta_dist_list / delta_dist_list.sum()
    delta_dist_norm_list = delta_dist_norm.tolist()
    
    for i in range(len(model.model.layers)):
        print(f"Hidden State Drift (Layer #{i}): {delta_dist_norm_list[i]:.4f}")
    print("\n")

    Lw = []
    quantile = args.delta
    threshold = torch.quantile(delta_dist_norm, quantile).item()

    for l in range(len(delta_dist_norm_list)):
        if delta_dist_norm_list[l] <= threshold:
            Lw.append(l)

    print(f"Detected Sparse Layer Lw: {Lw if Lw is not None else 'Not found'}")

def main():
    parser = argparse.ArgumentParser()
    # -----------------Default setting------------------------------------
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="model path")
    # -----------------Profile setting---------------------------------
    parser.add_argument("--delta", type=float, default=0.5, help="delta for the sparse attention.")
    parser.add_argument("--context_length", type=int, default=131072, help="context length for evaluation.")
    # ------------------------------------------------------------------- 

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map='auto', use_fast=False, legacy=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cpu', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=True, trust_remote_code=True)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False

    if args.model_path == "meta-llama/Llama-3.1-8B-Instruct":
        from sparse_attn.token_sparse.profile.monkeypatch import replace_llama
        replace_llama(model, args)
    elif args.model_path == "mistralai/Mistral-Nemo-Instruct-2407":
        from sparse_attn.token_sparse.profile.monkeypatch import replace_mistral
        replace_mistral(model, args)

    # Evaluation
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map) 
        
    benchmark(model, tokenizer, args)
    
if __name__ == "__main__":
    main()
