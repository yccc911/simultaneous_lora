from typing import Dict
from mlora.model_llama import LlamaModel
from transformers import LlamaForCausalLM
import os
import json
import torch
import logging

FORMAT = '%(asctime)s %(filename)s %(module)s %(funcName)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

# convert huggingface model to pytorch model
def convert_hf_to_pth(source: str, dest: str):
    src_model = LlamaForCausalLM.from_pretrained(source)
    # src_model.eval()
    torch.save(src_model.state_dict(), dest)


# save lora model
def save_lora_model(model: LlamaModel, config: Dict[str, str], dir_suffix=""):

    logging.info("Saving general lora model")
    general_lora = config['general_lora']
    lora_name = general_lora["name"]
    lora_output_dir = general_lora["output"]
    if dir_suffix != "":
        lora_output_dir += os.sep + general_lora["output"] + "_" + dir_suffix

    if not os.path.exists(lora_output_dir):
        os.makedirs(lora_output_dir)

    lora_weight_dict, target_modules = model.get_lora_weight_dict(lora_name)

    torch.save(lora_weight_dict, lora_output_dir + os.sep + "adapter_model.bin")

    adapter_config = {}
    adapter_config["lora_alpha"] = general_lora["alpha"]
    adapter_config["lora_dropout"] = general_lora["dropout"]
    adapter_config["r"] = general_lora["r"]
    adapter_config["peft_type"] = "LORA"
    adapter_config["task_type"] = "CAUSAL_LM"
    adapter_config["bias"] = "none"
    adapter_config["target_modules"] = target_modules

    with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=4)

    for lora_config in config["lora"]:
        lora_name = lora_config["name"]
        logging.info(f"Saving indepedent lora model: {lora_name}")
        lora_output_dir = lora_config["output"]
        if dir_suffix != "":
            lora_output_dir += os.sep + lora_config["output"] + "_" + dir_suffix

        if not os.path.exists(lora_output_dir):
            os.makedirs(lora_output_dir)

        lora_weight_dict, target_modules = model.get_lora_weight_dict(lora_name)

        torch.save(lora_weight_dict, lora_output_dir + os.sep + "adapter_model.bin")

        adapter_config = {}
        adapter_config["lora_alpha"] = general_lora["alpha"]
        adapter_config["lora_dropout"] = lora_config["dropout"]
        adapter_config["r"] = general_lora["r"]
        adapter_config["peft_type"] = "LORA"
        adapter_config["task_type"] = "CAUSAL_LM"
        adapter_config["bias"] = "none"
        adapter_config["target_modules"] = target_modules

        with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)
