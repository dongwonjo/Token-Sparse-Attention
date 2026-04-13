#!/usr/bin/env bash

# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Download
cd eval/ruler/data/synthetic/json
# download Paul Graham Essays for the needle test
python download_paulgraham_essay.py
# download SQuAD and HotpotQA for the QA test
bash download_qa_dataset.sh
cd ../../../../../

# donwload nltk
python -c "import nltk; nltk.download('punkt')"
python -c "import nltk; nltk.download('punkt_tab')"

# Prepare data
SEQ_LENGTHS=(
    4096
    8192
    16384
    32768
    65536
    131072
)

TASKS=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL_PATH>"
    echo "Example: $0 meta-llama/Llama-3.1-8B-Instruct"
    exit 1
fi

MODEL_PATH="$1"
NUM_SAMPLES=32
TOKENIZER_TYPE="hf"

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        python -m eval.ruler.data.prepare \
            --save_dir "eval/ruler/data/models" \
            --benchmark "synthetic" \
            --task ${TASK} \
            --tokenizer_path $MODEL_PATH \
            --tokenizer_type $TOKENIZER_TYPE \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --num_samples ${NUM_SAMPLES}
    done
done