import torch
import logging
import sys
import os
import time
import glob
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForCausalLM
from sparse_attn.monkeypatch import replace_llama, replace_mistral


def load_model(args):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, device_map='auto', use_fast=False, legacy=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cpu', torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=True, trust_remote_code=True)
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False

    if args.model_path == "meta-llama/Llama-3.1-8B-Instruct":
        replace_llama(model, args)
    elif args.model_path == "mistralai/Mistral-Nemo-Instruct-2407":
        replace_mistral(model, args)
        
    return model, tokenizer


def generate_prompt(tokenizer, target_len):
    needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    question = "What is the best thing to do in San Francisco?"
    context = ""

    while len(tokenizer.encode(context, add_special_tokens=False)) < target_len:
        for file in glob.glob(f"eval/needle/PaulGrahamEssays/*.txt"):
            with open(file, 'r') as f:
                context += f.read()
                
    tokens = tokenizer.encode(context, add_special_tokens=False)
    if len(tokens) > target_len:
        context = tokenizer.decode(tokens[:target_len], skip_special_tokens=True)

    tokens_needle = tokenizer.encode(needle, add_special_tokens=False)
    tokens_context = tokenizer.encode(context, add_special_tokens=False)

    if len(tokens_context) + len(tokens_needle) > target_len:
        tokens_context = tokens_context[:target_len - len(tokens_needle)]       

    depth_percent = 50
    insertion_point = int(len(tokens_context) * (depth_percent / 100))
    tokens_new_context = tokens_context[:insertion_point]
    period_tokens = tokenizer.encode('.', add_special_tokens=False)

    while tokens_new_context and tokens_new_context[-1] not in period_tokens:
        insertion_point -= 1
        tokens_new_context = tokens_context[:insertion_point]
    
    tokens_new_context += tokens_needle + tokens_context[insertion_point:]
    new_context = tokenizer.decode(tokens_new_context, skip_special_tokens=True)
    context=f"<|im_start|> This is a very long story book: <book> {new_context} </book>.\n Based on the content of the book, Question: {question}\nAnswer:"
    
    return context


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
