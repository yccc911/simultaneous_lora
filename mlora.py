# m-LoRA: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
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
#
# Copyright (C) 2023 All Rights Reserved.
#
# Github:  https://github.com/TUDB-Labs/multi-lora-fine-tune

import json
import torch
import mlora
import random
import argparse
import logging
from typing import Dict, Tuple, List
from tqdm.auto import tqdm

# Command Line Arguments
parser = argparse.ArgumentParser(description='m-LoRA main program')
parser.add_argument('--base_model', type=str,
                    help='Path to or name of base model')
parser.add_argument('--model_type', type=str, default="llama",
                    help='The model type, support: llama, chatglm')
parser.add_argument('--inference', action="store_true",
                    help='The inference mode (just for test)')
parser.add_argument('--load_lora', action="store_true",
                    help="Load lora from file instead of init randomly")
parser.add_argument('--disable_lora', action="store_true",
                    help="Disable the lora modules")
parser.add_argument('--tokenizer', type=str,
                    help='Path to or name of tokenizer')
parser.add_argument('--load_8bit', action="store_true",
                    help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load model in 4bit mode')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
parser.add_argument('--config', type=str,
                    help='Path to finetune configuration')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed in integer, default is 42')
parser.add_argument('--log', type=bool, default=True,
                    help='Turn on or off log, default is true')

args = parser.parse_args()


if torch.cuda.is_available():
    logging.info('NVIDIA CUDA initialized successfully.')
    logging.info('Total %i GPU(s) detected.' % torch.cuda.device_count())
else:
    print('m-LoRA requires NVIDIA CUDA computing capacity. Please check your PyTorch installation.')
    exit(-1)


if args.base_model is None:
    print('error: Argument --base_model are required.')
    parser.print_help()
    exit(-1)


if args.config is None:
    print('error: Argument --config are required.')
    parser.print_help()
    exit(-1)

FORMAT = '%(asctime)s %(filename)s %(module)s %(funcName)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

# Functions
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def load_base_model() -> Tuple[mlora.Tokenizer, mlora.LLMModel]:
    if args.model_type == "llama":
        model = mlora.LlamaModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            # log_fn=log
        )
    else:
        raise ValueError(f"unknown model type {args.model_type}")

    logging.info("Initializing tokenizer")
    tokenizer = mlora.Tokenizer(args.base_model)

    model.pad_token_id_ = tokenizer.pad_id_

    return tokenizer, model


# 1 general lora & multiple lora models
def init_lora_model(config: Dict[str, any], llm_model: mlora.LLMModel):
    if args.disable_lora:
        return

    general_lora = config['general_lora']
    lora_weight = None
    if args.load_lora:
        adapter_file_path = general_lora["output"] + "/adapter_model.bin"
        print(f"load {adapter_file_path}")
        lora_weight = torch.load(adapter_file_path)

    logging.info('Initializing LoRA: general')
    llm_model.init_lora_weight(general_lora["name"],
                            general_lora["r"],
                            general_lora["alpha"],
                            general_lora["dropout"],
                            general_lora["target_modules"],
                            lora_weight)

    for lora_config in config["lora"]:
        lora_weight = None
        if args.load_lora:
            adapter_file_path = lora_config["output"] + "/adapter_model.bin"
            print(f"load {adapter_file_path}")
            lora_weight = torch.load(adapter_file_path)
        logging.info(f'Initializing LoRA: {lora_config["name"]}')
        llm_model.init_lora_weight(lora_config["name"],
                                    general_lora["r"],
                                    general_lora["alpha"],
                                    lora_config["dropout"],
                                    lora_config["target_modules"],
                                    lora_weight)
    logging.info(f"Initialized lora: {llm_model.layers_[-1].wq_.loras_.keys()}")


def get_optimizer(config: Dict[str, any], train_paramas: Dict[str, torch.Tensor]) -> Dict[str, torch.optim.Optimizer]:
    optimizer: Dict[str, torch.optim.Optimizer] = {}

    for lora_config in config["lora"]:
        adapter_name = lora_config["name"]
        optim_name = lora_config["optim"]
        lr = lora_config["lr"]
        if optim_name == "sgd":
            momentum = 0
            if "momentum" in lora_config:
                momentum = lora_config["momentum"]
            optimizer[adapter_name] = (torch.optim.SGD(train_paramas[adapter_name], lr=lr, momentum=momentum))
        elif optim_name == "adamw":
            optimizer[adapter_name] = (torch.optim.AdamW(train_paramas[adapter_name], lr=lr))
        else:
            raise ValueError(f"unknown optimizer {optim_name}")

    return optimizer


def get_general_optimizer(config: Dict[str, any], general_train_para: torch.Tensor) -> torch.optim.Optimizer:
    general_optimizer: torch.optim.Optimizer = None

    general_lora = config['general_lora']
    optim_name = general_lora["optim"]
    lr = general_lora["lr"]

    if optim_name == "sgd":
        momentum = 0
        if "momentum" in general_lora:
            momentum = general_lora["momentum"]
        general_optimizer = (torch.optim.SGD(general_train_para, lr=lr, momentum=momentum))
    elif optim_name == "adamw":
        general_optimizer = (torch.optim.AdamW(general_train_para, lr=lr))
    else:
        raise ValueError(f"unknown optimizer {optim_name}")

    return general_optimizer


# (?) should be in accordance with the number of training inputs from every dataset
def get_accumulation_steps(config: Dict[str, any]) -> int:
    general_lora = config['general_lora']
    batch_size = general_lora["batch_size"]
    micro_batch_size = general_lora["micro_batch_size"]

    if batch_size < micro_batch_size or batch_size % micro_batch_size != 0:
        raise ValueError(f"error batch_size {batch_size} and micro batch size {micro_batch_size}")

    ret_accumulation_step = batch_size / micro_batch_size
    return ret_accumulation_step


# to get test result and want early stop it
def train(config: Dict[str, any], llm_model: mlora.LLMModel, dispatcher: mlora.Dispatcher):
    logging.debug("Getting training parameters for every independent lora model")
    all_train_paramas: Dict[str, List[torch.Tensor]] = llm_model.get_train_paramas(config)
    logging.debug("Getting optimizers for every independent lora model")
    all_optimizer: Dict[str, torch.optim.Optimizer] = get_optimizer(config, all_train_paramas)

    logging.debug("Getting training parameters for general lora model")
    general_train_para: torch.Tensor = llm_model.get_general_train_paramas()
    logging.debug("Getting optimizers for general lora model")
    general_optimizer: torch.optim.Optimizer = get_general_optimizer(config, general_train_para)

    accumulation_step: int = get_accumulation_steps(config)

    loss_fn = torch.nn.CrossEntropyLoss()

    step_cnt = {
        "general_lora": 0
    }
    for lora in config['lora']:
        step_cnt[lora['name']] = 0

    progress = tqdm(total=dispatcher.get_total_train_data_len() // config['general_lora']['micro_batch_size'], desc="Training...", leave=True, position=0)
    logging.info("Start training!")
    while not dispatcher.check_task_done():
        input: mlora.LoraBatchData = dispatcher.get_train_data()

        logging.debug("LLMModel forwarding...")
        output = llm_model.forward(input)
        labels = torch.tensor(input.batch_tokens_, dtype=torch.long).to(args.device)

        loss_input = output[..., :-1, :].contiguous().view(-1, llm_model.vocab_size_)
        loss_target = labels[..., 1:].contiguous().view(-1)
        loss = loss_fn(loss_input, loss_target)
        logging.debug(f"adapter: {input.adapter_name_} step: {step_cnt[input.adapter_name_]} loss: {loss}")
        progress.set_postfix({"adapter": input.adapter_name_, "step": step_cnt[input.adapter_name_], "loss": loss.item()})
        progress.update(1)
        loss /= accumulation_step

        step_cnt['general_lora'] += 1
        step_cnt[input.adapter_name_] += 1

        # logging.info("loss.backward: calculating gradients")
        loss.backward()
        if step_cnt[input.adapter_name_] % accumulation_step == 0:
            logging.debug(f"Adapter-{input.adapter_name_} gradient updates")
            all_optimizer[input.adapter_name_].step()
            all_optimizer[input.adapter_name_].zero_grad()
        if step_cnt['general_lora'] % accumulation_step == 0:
            logging.debug(f"Adapter-general gradient updates")
            general_optimizer.step()
            general_optimizer.zero_grad()

        if step_cnt['general_lora'] % config["save_step"] == 0:
            logging.info(f"step: {step_cnt[input.adapter_name_]} saving model")
            mlora.save_lora_model(llm_model, config, f"{step_cnt}")

    mlora.save_lora_model(llm_model, config)


# def construct(instance):
#     instance['prompt'] = f"{instance['sentence']}\n\nQuestion: {instance['question']}\n\nAnswer: {instance['answer']}.\n\nIs the answer a plausible one, yes or no?\n\n"
#     return instance

# def inference(lora_config: Dict[str, any], llm_model: mlora.LLMModel, tokenizer: mlora.Tokenizer):

#     target_lora = lora_config['name']
#     inference_max_len = 1024

#     from datasets import load_dataset
#     test_data = load_dataset("mc_taco", split="test")
#     test_data = test_data.map(construct)
#     prompt_list = []

#     ans_list = []
#     # for i in range(0, len(test_data), 10):
#     #   prompt_list.append(test_data[i]['prompt'])
#     #   ans_list.append(test_data[i]['label'])
#     for data in test_data:
#         prompt_list.append(data['prompt'])
#         ans_list.append(data['label'])

#     for input_raw in prompt_list:

#         tokens = tokenizer.encode(input_raw, True, False)
#         token_len = len(tokens)
#         while len(tokens) < inference_max_len:
#             tokens.append(tokenizer.pad_id_)

#         input_data = mlora.LoraBatchData(
#             prompts_=[input_raw],
#             adapter_name_=target_lora,
#             batch_tokens_=[tokens],
#             tokens_len_without_pad_=[token_len],
#             batch_seq_len_=inference_max_len,
#             expand_side_=[lora_config['expand_side']],
#             inference_model_=True)

#         eos_flag: bool= False
#         for pos in range(token_len, inference_max_len):
#             with torch.no_grad():
#                 # batch_size, seq_len, voc_logs
#                 outputs = llm_model.forward(input_data)
#                 next_token = outputs[:, pos - 1, :]
#                 next_token = torch.argmax(next_token, dim=-1)
#                 for idx in range(len(input_data.batch_tokens_)):
#                     input_data.batch_tokens_[idx][pos] = next_token[idx].item()
#                     # end of the sentence
#                     if next_token[idx].item() == tokenizer.eos_id_:
#                         eos_flag[idx] = True
#                     input_data.tokens_len_without_pad_[idx] = input_data.tokens_len_without_pad_[idx] + 1
#             # check if the sentence ends
#             if eos_flag:
#                 break

#         for idx, output in enumerate(input_data.batch_tokens_):
#             print(f"# LORA{idx} OUTPUT IS:")
#             print(tokenizer.decode(output))


# Main Function
if __name__ == "__main__":
    setup_seed(args.seed)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model()
    init_lora_model(config, model)

    torch.cuda.empty_cache()

    # if args.inference:
    #     inference(config, model, tokenizer)
    # else:
    dispatcher = mlora.Dispatcher(config, tokenizer)
    train(config, model, dispatcher)
