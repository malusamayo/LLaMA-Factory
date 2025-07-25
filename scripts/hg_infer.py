# Copyright 2025 the LlamaFactory team.
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

import argparse
import gc
import json
from typing import Optional

from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer
from transformers import pipeline
import torch
import json

def main(args):
    batch_size = 64

    pipe = pipeline(
        "image-text-to-text",
        model=args.model_name_or_path,
        device_map="auto",
        batch_size=32,
        torch_dtype=torch.bfloat16
    )

    with open(f"/home/ychenyang/LLaMA-Factory/data/{args.file_name}", "r") as f:
        df_test = json.loads(f.read())

    # df_test = df_test[:30]

    messages = [[
        {
            "role": "user",
            "content": [
                {"type": "image", "url": line['images'][0]},
                {"type": "text", "text": line['conversations'][0]['value']}
            ]
        } 
    ] for line in df_test]


    outputs = []
    for start_idx in tqdm(range(0, len(messages), batch_size)):
        outputs += pipe(text=messages[start_idx: start_idx+batch_size], max_new_tokens=512)
    
    all_prompts = [
        line['conversations'][0]['value']
        for line in df_test
    ]
    all_labels = [
        line['conversations'][1]['value']
        for line in df_test
    ]

    all_preds = [output[0]["generated_text"][-1]["content"] for output in outputs]

    with open(args.save_name, "w", encoding="utf-8") as f:
        for text, pred, label in zip(all_prompts, all_preds, all_labels):
            f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image-text-to-text pipeline")
    parser.add_argument('--model_name_or_path', type=str, help='Path to the model')
    parser.add_argument('--save_name', type=str, help='Name of the output file')
    parser.add_argument('--file_name', type=str, help='Name of the input file')
    args = parser.parse_args()

    # model_name = '/home/ychenyang/LLaMA-Factory/saves/gemma-3-4b-it/full/sft/checkpoint-2532'
    # save_name = 'generated_predictions_gemma-3-4b.jsonl'
    # file_name = 'oss_chameleon_test.json'
    main(args)