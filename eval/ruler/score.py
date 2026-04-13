import os
import json
import csv
import re
from tqdm import tqdm

from eval.ruler.eval.synthetic.constants import (
    string_match_all, 
    string_match_part)

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

DATASET_TO_METRIC = {
    "niah_single_1": string_match_all,
    "niah_single_2": string_match_all,
    "niah_single_3": string_match_all,
    "niah_multikey_1": string_match_all,
    "niah_multikey_2": string_match_all,
    "niah_multikey_3": string_match_all,
    "niah_multiquery": string_match_all,
    "niah_multivalue": string_match_all,
    "cwe": string_match_all,
    "fwe": string_match_all,
    "vt": string_match_all,
    "qa_1": string_match_part,
    "qa_2": string_match_part,
}

results_list = [
    ["Context Length", "4096", "8192", "16384", "32768", "65536", "131072"],
    ["Results"]
    ]


def postprocess_pred(predict_str: str):
    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r"[\x00-\x1f]")
    predict_str = np_pattern.sub("\n", predict_str).strip()

    return predict_str

def cal_score(save_path):
    context_length_files = os.listdir(save_path)
    all_scores = dict()
    for context_length_file in context_length_files:
        scores = dict()
        context_length_file_path = os.path.join(save_path, context_length_file)
        task_files = os.listdir(context_length_file_path)
        for task_file in task_files:
            if not task_file.endswith("jsonl"):
                continue
            task = task_file.split('.')[0]
            task_file_path = os.path.join(save_path, context_length_file, task_file)
            predictions, answers, lengths = [], [], []

            with open(task_file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc=f"Evaluation on {context_length_file}-{task}..."):
                    data = json.loads(line)
                    predictions.append(postprocess_pred(data["pred"]))
                    answers.append(data["answers"])
                    if "length" in data:
                        lengths.append(data["length"])
    
            score = DATASET_TO_METRIC[task](predictions, answers)
            scores[task] = score
        
        if len(scores.keys()) == len(TASKS):
            scores["average"] = f"{sum(scores.values()) / len(scores.values()):.2f}"
            all_scores[context_length_file] = scores["average"]

        with open(os.path.join(context_length_file_path, "results.json"), "w") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        
    if len(all_scores.keys()) == (len(results_list[0]) - 1):
        for i in range((len(results_list[0]) - 1)):
            results_list[1].append(all_scores[results_list[0][i+1]])
        with open(os.path.join(save_path, f"results.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(results_list)