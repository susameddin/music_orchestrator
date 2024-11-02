import torch
from peft import set_peft_model_state_dict, get_peft_model_state_dict

def save_lora(lora_model, path):
    peft_state_dict = get_peft_model_state_dict(lora_model)
    torch.save(peft_state_dict, path)

def load_lora(lora_model, path):
    peft_state_dict = torch.load(path)
    result = set_peft_model_state_dict(lora_model, peft_state_dict)