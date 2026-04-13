import sys
sys.path.append(".")

import os
import json
from tqdm import tqdm
from pathlib import Path
import subprocess

import torch
from typing import List

from eval.ruler.score import cal_score

TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

# Reference: eval/ruler/data/synthetic/constants.py
TASK_TO_MAX_NEW_TOKNES = {
    "niah_single_1": 128,
    "niah_single_2": 128,
    "niah_single_3": 128,
    "niah_multikey_1": 128,
    "niah_multikey_2": 128,
    "niah_multikey_3": 128,
    "niah_multiquery": 128,
    "niah_multivalue": 128,
    "cwe": 120,
    "fwe": 50,
    "vt": 30,
    "qa_1": 32,
    "qa_2": 32,
}

MODEL_TO_MAX_LENGTHS = {
    "meta-llama/Llama-2-7b-chat-hf": 4096,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8192,
    "meta-llama/Llama-3.1-8B-Instruct": 131072,
    "meta-llama/Llama-3.2-1B-Instruct": 131072,
    "meta-llama/Llama-3.2-3B-Instruct": 131072,
    "mistralai/Mistral-Nemo-Instruct-2407": 131072,
    "mistralai/Ministral-8B-Instruct-2410": 131072,
    "Qwen/Qwen2.5-7B-Instruct": 131072,
}


@torch.inference_mode()
def generate(model, tokenizer, data_path, out_path, max_gen, max_length, args, logger):
    length_list = []
    index_list = []
    inputs_list = []
    outputs_list: List[List[str]] = [] # List of List
    device = model.device

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            length_list.append(example["length"])
            index_list.append(example["index"])
            outputs_list.append(example["outputs"])
            inputs_list.append(example["input"])

  
    for i in tqdm(range(len(inputs_list))):
        prompt = inputs_list[i]
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=True).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        input = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=True).to(device)
        context_length = input.input_ids.shape[-1]
        
        with torch.no_grad():
            output = model.generate(**input, num_beams=1, do_sample=False, temperature=1.0, top_p=1.0, 
                                    min_length=context_length+1, max_new_tokens=max_gen, eos_token_id=[tokenizer.eos_token_id])[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": outputs_list[i], "input": inputs_list[i], 
                       "length": length_list[i], "index": index_list[i]}, f, ensure_ascii=False)
            f.write('\n')


def evaluate(model, tokenizer, args, logger):
    save_dir = os.path.join(f"{args.save_dir}", "ruler")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Dataset Check
    model_name = args.model_path.split('/')[-1]
    dataset_path = f"eval/ruler/data/models/{model_name}"
    CONTEXT_LENGTHS = args.ruler_length
    
    def check_dataset():
        missing = []
        for context_length in CONTEXT_LENGTHS:
            for task in TASKS:
                data_path = os.path.join(
                    dataset_path, str(context_length), task, "validation.jsonl"
                )
                if not os.path.exists(data_path):
                    missing.append(data_path)
        return missing

    missing_files = check_dataset()

    if len(missing_files) > 0:
        logger.warning(
            f"[RULER] Missing {len(missing_files)} files. Running download_dataset.sh"
        )
        
        script_path = "./eval/ruler/download_dataset.sh"
        try:
            subprocess.run(
                ["bash", "./eval/ruler/download_dataset.sh", args.model_path],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to run {script_path}"
            ) from e

        
        missing_files = check_dataset()
        if len(missing_files) > 0:
            raise ValueError(
                f"RULER data still missing {len(missing_files)} files after download."
            )
    else:
        logger.info(f"RULER data found. Base path: {dataset_path}")

    total_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for idx_L, context_length in enumerate(CONTEXT_LENGTHS):
        for idx_D, task in enumerate(TASKS):
            iteration = (idx_L) * len(TASKS) + (idx_D+1)
            max_gen = TASK_TO_MAX_NEW_TOKNES[task]
            max_length = MODEL_TO_MAX_LENGTHS[args.model_path]
            data_path = os.path.join(
                    dataset_path, str(context_length), task, "validation.jsonl")

            Path(os.path.join(save_dir, str(context_length))).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(save_dir, str(context_length), f"{task}.jsonl")

            start.record()
            logger.info(f"\nGenerating responses...\nContext Length: {context_length} | Task: {task} | ({(iteration)}/{len(CONTEXT_LENGTHS) * len(TASKS)})\n")
            generate(model=model, tokenizer=tokenizer, data_path=data_path, out_path=out_path, 
                     max_gen=max_gen, max_length=max_length, args=args, logger=logger)
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)

            logger.info(f"\nCalculating scores for {context_length}-{task}\n")
            cal_score(save_path=save_dir)
    
    logger.info(f"\nEvaluation Total Time: {total_time/1000:.2f} sec\n")
