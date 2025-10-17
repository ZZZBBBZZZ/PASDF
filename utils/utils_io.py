import os
import random
import numpy as np
import torch
import yaml

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

def read_params(setting_path):
    """Read the settings from the settings.yaml file. These are the settings used during training."""
    # training_settings_path = os.path.join('/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results/Real3D_AD/runs_sdf' , cfg['folder_sdf'], 'settings.yaml')
    training_settings_path = os.path.join(setting_path)
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)

    return training_settings