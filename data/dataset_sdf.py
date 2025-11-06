import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]
import results
import trimesh
import pysdf
results_path = os.path.join(current_dir, "../results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SDFDataset(Dataset):
    """
    TODO: adapting to handle multiple objects
    """
    def __init__(self, dataset_name,class_idx):
        samples_dict = np.load(os.path.join(results_path, dataset_name,f'samples_dict_{dataset_name}.npy'), allow_pickle=True).item()
        self.data = dict()
        for obj_idx in list(samples_dict.keys()): 
            if obj_idx == class_idx:
                for key in samples_dict[obj_idx].keys():  
                    value = torch.from_numpy(samples_dict[obj_idx][key]).float().to(device)
                    if len(value.shape) == 1:    
                        value = value.view(-1, 1)
                    if key not in list(self.data.keys()):
                        self.data[key] = value
                    else:
                        self.data[key] = torch.vstack((self.data[key], value))
        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf

if __name__=='__main__':
    dataset_name = "ShapeNetCore"
    dataset = SDFDataset(dataset_name)
