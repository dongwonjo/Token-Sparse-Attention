import sys
sys.path.append(".")

import json
import os
from pathlib import Path
from tqdm import tqdm
import torch

from datasets import load_dataset, Features, Value, Sequence

from eval.infinite_bench.score import cal_score
from eval.infinite_bench.eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    MODEL_TO_MAX_LENGTH,
    create_prompt,
    dump_jsonl,
    get_answer,
)

def build_chat(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    
    return prompt

@torch.inference_mode()
def generate(data, dataset, max_length, max_gen, model, tokenizer, out_path, args, logger):
    preds = []
    for i, eg in enumerate(tqdm(data, desc="Generating Responses...")):
        prompt = create_prompt(eg, dataset)
        ground_truth = get_answer(eg, dataset)

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if args.model_path == "meta-llama/Llama-3.1-8B-Instruct":
            prompt = build_chat(tokenizer, prompt)
        
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]
        
        with torch.no_grad():
            output = model.generate(**input, num_beams=1, do_sample=False, temperature=1.0, top_p=1.0, min_length=context_length+1, max_new_tokens=max_gen)[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True).strip()
            
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": ground_truth,
            }
        )
    dump_jsonl(preds, out_path)
    torch.cuda.empty_cache()
    
def evaluate(model, tokenizer, args, logger):
    save_dir = os.path.join(f"{args.save_dir}", "infinite_bench")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    max_length = MODEL_TO_MAX_LENGTH[args.model_path]
    dataset_list = ["kv_retrieval", "number_string", "passkey", \
                    "math_find", "code_debug", \
                    "longbook_choice_eng", "longbook_qa_eng", "longdialogue_qa_eng"]

    total_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    for idx, dataset in enumerate(args.infinite_bench_dataset):
        if dataset not in dataset_list:
            raise ValueError(f"Dataset {dataset} not found in datasets")

        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[dataset]
        ft = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string"))
        })
        data = load_dataset('xinrongzhang2022/InfiniteBench', split=dataset, features=ft)
        
        out_path = os.path.join(save_dir, f"{dataset}.jsonl")

        # Generation
        start.record()
        logger.info(f"\nGenerating responses for {dataset} ({(idx+1)}/{len(args.infinite_bench_dataset)})\n")
        generate(data=data, dataset=dataset, max_length=max_length, max_gen=max_new_tokens, 
                 model=model, tokenizer=tokenizer, out_path=out_path, args=args, logger=logger)
        end.record()
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)

        # Evaluation
        logger.info(f"\nCalculating scores for {dataset} ({(idx+1)}/{len(args.infinite_bench_dataset)})\n")
        cal_score(save_path=save_dir)
    
    logger.info(f"\nEvaluation Total Time: {total_time/1000:.2f} sec\n")