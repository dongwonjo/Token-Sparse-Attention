import os
import random
import numpy as np
import torch
import argparse
import pprint
import time
from pathlib import Path
from transformers import set_seed
from accelerate import infer_auto_device_map, dispatch_model

from utils import utils

torch.backends.cudnn.benchmark = True

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
    # -----------------Needle setting------------------------------------
    parser.add_argument("--eval_needle", action="store_true", default=False)
    parser.add_argument('-s', '--s_len', default=0, metavar='N', type=int)
    parser.add_argument('-e', '--e_len', default=128000, metavar='N', type=int)
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument("--needle_length", nargs='+', type=int, 
                        default=[8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000, 72000, 80000, 88000, 96000, 104000, 112000, 120000, 128000])
    parser.add_argument("--needle", type=str, default="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n")
    parser.add_argument("--retrieval_question", type=str, default="What is the best thing to do in San Francisco?")
    parser.add_argument("--expected_answer", type=str, default="eat a sandwich and sit in Dolores Park on a sunny day.", help="Path to save the output")
    # -----------------LongBench setting----------------------------------
    parser.add_argument("--eval_longbench", action="store_true", default=False)
    parser.add_argument("--longbench_type", type=str, default="longbench", choices=["longbench", "longbench-e"])
    parser.add_argument("--longbench_dataset", nargs='+', type=str, 
                        default=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", 
                                 "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", 
                                 "passage_count", "passage_retrieval_en", "lcc", "repobench-p"])
    # -----------------Ruler setting--------------------------------------
    parser.add_argument("--eval_ruler", action="store_true", default=False)
    parser.add_argument("--ruler_length", nargs='+', type=int, 
                        default=[4096, 8192, 16384, 32768, 65536, 131072])  
    # -----------------InfiniteBench setting------------------------------
    parser.add_argument("--eval_infinite_bench", action="store_true", default=False)
    parser.add_argument('--infinite_bench_dataset', nargs='+', type=str, 
                        default=["kv_retrieval", "number_string", "passkey", "math_find", "code_debug", \
                                 "longbook_choice_eng", "longbook_qa_eng", "longdialogue_qa_eng"])
    # --------------------------------------------------------------------
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    set_seed(args.seed)
        
    # Init logger
    model_name = args.model_path.split('/')[-1]
    if args.save_dir:
        tm = time.localtime(time.time())
        f_name = f"{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}"
        args.save_dir = os.path.join(args.save_dir, model_name, f_name)
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
    logger = utils.create_logger(args.save_dir)
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)

    # Load model
    model, tokenizer = utils.load_model(args)
    
    # Evaluation
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, no_split_module_classes=[block_class_name])
    model = dispatch_model(model, device_map=device_map) 

    if args.eval_needle:
        from eval.needle.evaluate import evaluate
        evaluate(model, tokenizer, args, logger)
    
    if args.eval_longbench:
        from eval.longbench.evaluate import evaluate
        evaluate(model, tokenizer, args, logger)

    if args.eval_ruler:
        from eval.ruler.evaluate import evaluate
        evaluate(model, tokenizer, args, logger)

    if args.eval_infinite_bench:
        from eval.infinite_bench.evaluate import evaluate
        evaluate(model, tokenizer, args, logger)
    
if __name__ == "__main__":
    main()
