#!/usr/bin/env bash

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

set -e

if [ ! -f squad.json ]; then
    echo "[QA] Downloading SQuAD..."
    wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json
else
    echo "[QA] squad.json already exists, skip"
fi

if [ ! -f hotpotqa.json ]; then
    echo "[QA] Downloading HotpotQA..."
    wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json -O hotpotqa.json
else
    echo "[QA] hotpotqa.json already exists, skip"
fi
