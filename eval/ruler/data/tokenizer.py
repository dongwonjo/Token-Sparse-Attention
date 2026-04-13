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


import os
from typing import List

def select_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type == "nemo":
        return NeMoSentencePieceTokenizer(model_path=tokenizer_path)
    elif tokenizer_type == "hf":
        return HFTokenizer(model_path=tokenizer_path)
    elif tokenizer_type == "openai":
        raise ValueError(f"Unknown tokenizer_type {tokenizer_type}")
    elif tokenizer_type == "gemini":
        raise ValueError(f"Unknown tokenizer_type {tokenizer_type}")
    else:
        raise ValueError(f"Unknown tokenizer_type {tokenizer_type}")


class NeMoSentencePieceTokenizer:
    """
    Tokenizer from NeMo SentencePieceTokenizer
    """

    def __init__(self, model_path) -> None:
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import (
            SentencePieceTokenizer,
        )

        self.tokenizer = SentencePieceTokenizer(model_path=model_path)

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.text_to_tokens(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.tokens_to_text(tokens)
        return text


class HFTokenizer:
    """
    Tokenizer from HF models
    """

    def __init__(self, model_path) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text


