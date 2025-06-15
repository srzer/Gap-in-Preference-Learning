import torch
import os
def load_and_combine_state_dicts(folder_path, map_location):
    combined_state_dict = {}
    files = os.listdir(folder_path)
    for file in files:
        if file.startswith("pytorch") and file.endswith(".bin"):
            state_dict = torch.load(os.path.join(folder_path, file), map_location=map_location)
            combined_state_dict.update(state_dict)
    return combined_state_dict
  
def load_and_combine_lora_state_dicts(folder_path, map_location):
    combined_state_dict = {}
    files = os.listdir(folder_path)
    for file in files:
        if file.startswith("pytorch") and file.endswith(".bin"):
            state_dict = torch.load(os.path.join(folder_path, file), map_location=map_location)
            lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
            combined_state_dict.update(lora_state_dict)
    return combined_state_dict