import os
import random
import numpy as np
import torch
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
    context = generate_prompt(tokenizer, target_len=args.context_length)
    prompt = tokenizer(context, return_tensors="pt")
    input_ids = prompt['input_ids'].to(model.device)
    attn_mask = prompt['attention_mask'].to(model.device)

    # Warmup
    if args.num_warmups > 0:
        for i in tqdm(range(args.num_warmups), desc="Running Warmup"):
            with torch.no_grad():
                outputs = model(input_ids, attn_mask)
            del outputs

            utils.cleanup_memory()

    latency_list = []
    
    # Run
    for i in tqdm(range(args.num_runs), desc="Running Prefill"):
        total_time = 0
        
        with torch.no_grad():
            outputs = model(input_ids, attn_mask)

        latency = 0
        for layer in range(len(model.model.layers)):
            latency += model.model.layers[layer].self_attn.latency
        latency /= len(model.model.layers)
        total_time += latency
        del outputs

        utils.cleanup_memory()
        latency_list.append(total_time)
        
    mean_latency = sum(latency_list) / len(latency_list)
    
    if args.attn_method == "vanilla":
        attn_method = "FlashAttention"
    elif args.attn_method == "minference":
        attn_method = "Minference"
    elif args.attn_method == "flexprefill":
        attn_method = "FlexPrefill"
    elif args.attn_method == "xattention":
        attn_method = "XAttention"
    
    if args.token_sparse:
        attn_method += " w/ Token Sparse Attention"
    else:
        attn_method += " w/o Token Sparse Attention"
    
    print(f"--" * 30)
    print(f"Attention Method = {attn_method}")
    print(f"Context Length = {args.context_length}")
    print(f"Attention Latency = {(mean_latency):.4f} msec")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1000**2 / 1000:.2f} GB")
    print(f"--" * 30)

def main():
    parser = argparse.ArgumentParser()
    # -----------------Default setting------------------------------------
    parser.add_argument("--save_dir", default="outputs", type=str, help="direction of save file")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="model path")
    # -----------------Sparse setting-------------------------------------
    parser.add_argument("--attn_method", type=str, default="vanilla", choices=["vanilla", "minference", "flexprefill", "xattention"])
    # Token-Sparse Attention Setting
    parser.add_argument("--token_sparse", action="store_true", default=False)
    parser.add_argument("--sparse_layer", nargs='+', type=int, default=None, 
                        help="Sparse layer indices. None works for default setting (delta=0.5 in paper).")
    parser.add_argument("--coverage", type=float, default=0.005, 
                        help="Coverage ratio (tau in paper) for the Token Sparse Attention.")
    parser.add_argument("--min_tokens", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=7)
    # Minference Setting
    parser.add_argument("--adaptive_budget", type=float, default=0.1)
    # FlexPrefill Setting
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.1)
    # XAttention Setting
    parser.add_argument("--xattention_stride", type=int, default=16)
    parser.add_argument("--xattention_threshold", type=float, default=0.9)
    parser.add_argument("--xattention_chunk_size", type=int, default=None)
    parser.add_argument("--xattention_use_triton", action=argparse.BooleanOptionalAction, default=True)
    # -----------------Benchmark setting---------------------------------
    parser.add_argument("--num_warmups", type=int, default=1, help="")
    parser.add_argument("--num_runs", type=int, default=5, help="")
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
        from benchmark.monkeypatch import replace_llama
        replace_llama(model, args)
    elif args.model_path == "mistralai/Mistral-Nemo-Instruct-2407":
        from benchmark.monkeypatch import replace_mistral
        replace_mistral(model, args)

    # Evaluation
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map) 
        
    benchmark(model, tokenizer, args)
    
if __name__ == "__main__":
    main()
