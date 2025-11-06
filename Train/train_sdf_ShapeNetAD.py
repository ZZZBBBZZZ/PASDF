import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]
import torch
import model.model_sdf as sdf_model
import torch.optim as optim
import data.dataset_sdf as dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils.utils_deepsdf import SDFLoss_multishape

from datetime import datetime
import numpy as np
import time
from utils import utils_deepsdf
from torch.utils.tensorboard import SummaryWriter
import yaml
# import config_files

dataset_path = os.path.join(current_dir, "../data/ShapeNetAD")
results_path = os.path.join(current_dir, "../results")
results_runs_sdf_path = os.path.join(current_dir,"../results/ShapeNetAD/runs_sdf")
config_files_path = os.path.join(current_dir,"../config_files")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

embed_kwargs = {
    'include_input': True,
    'input_dims': 3,  
    'max_freq_log2': 4,  
    'num_freqs': 10,  
    'log_sampling': True, 
    'periodic_fns': [torch.sin, torch.cos],  
}

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class Trainer():
    def __init__(self, train_cfg):
        self.train_cfg = train_cfg

    def __call__(self):
        # directories
        # self.timestamp_run = datetime.now().strftime('%d_%m_%H%M%S')   # timestamp to use for logging data
        # # self.runs_dir = os.path.dirname(str(runs.__file__))               # directory fo all runs
        # self.run_dir = os.path.join(results_runs_sdf_path, self.timestamp_run)  # directory for this run
        self.run_dir = results_runs_sdf_path
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        # Logging
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_path = os.path.join(self.run_dir, 'settings.yaml')
        with open(self.log_path, 'w') as f:
            yaml.dump(self.train_cfg, f)

        # calculate num objects in samples_dictionary, wich is the number of keys
        samples_dict_path = os.path.join(results_path, self.train_cfg["dataset"],f'samples_dict_{self.train_cfg["dataset"]}.npy')
        samples_dict = np.load(samples_dict_path, allow_pickle=True).item()

        int2str_dict_path =  os.path.join(results_path, self.train_cfg["dataset"],'idx_int2str_dict.npy')  
        int2str_dict = np.load(int2str_dict_path, allow_pickle=True).item() 



        for class_idx in list(samples_dict.keys()):

            class_name = int2str_dict[class_idx]
            
            save_dir = os.path.join(self.run_dir, class_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            self.model = sdf_model.SDFModel_ofo_PoseEmbedder(
                    self.train_cfg['num_layers'], 
                    self.train_cfg['skip_connections'], 
                    inner_dim=self.train_cfg['inner_dim'],
                    PoseEmbedder_size = 60
                ).float().to(device)

            # define optimisers
            self.optimizer_model = optim.Adam(self.model.parameters(), lr=self.train_cfg['lr_model'], weight_decay=0)


            if self.train_cfg['lr_scheduler']:
                self.scheduler_model =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')

            print(f"Processing class {class_name} (class_idx={class_idx})")   

            # get data
            train_loader, val_loader = self.get_loaders(class_idx)
            self.results = {
                'best_latent_codes' : []
            }

            best_loss = 10000000000
            start = time.time()
            for epoch in range(self.train_cfg['epochs']):
                print(f'============================ {class_name}Epoch {epoch} ============================')
                self.epoch = epoch

                avg_train_loss = self.train(train_loader)

                with torch.no_grad():
                    avg_val_loss = self.validate(val_loader)

                    if avg_val_loss < best_loss:
                        best_loss = np.copy(avg_val_loss)
                        best_weights = self.model.state_dict()

                        optimizer_model_state = self.optimizer_model.state_dict()

                        torch.save(best_weights, os.path.join(save_dir, 'weights.pt'))
                        torch.save(optimizer_model_state, os.path.join(save_dir, 'optimizer_model_state.pt'))

                    if self.train_cfg['lr_scheduler']:
                        self.scheduler_model.step(avg_val_loss)

                        self.writer.add_scalar('Learning rate (model)', self.scheduler_model._last_lr[0], epoch)

            end = time.time()
            print(f'Time elapsed: {end - start} s')
            torch.cuda.empty_cache()

    def get_loaders(self,class_idx):
        data = dataset.SDFDataset(self.train_cfg['dataset'],class_idx)

        if self.train_cfg['clamp']:
            data.data['sdf'] = torch.clamp(data.data['sdf'], -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])

        train_size = int(0.85 * len(data))
        val_size = len(data) - train_size
        train_data, val_data = random_split(data, [train_size, val_size])
        train_loader = DataLoader(
                train_data,
                batch_size=self.train_cfg['batch_size'],
                shuffle=True,
                drop_last=True
            )
        val_loader = DataLoader(
            val_data,
            batch_size=self.train_cfg['batch_size'],
            shuffle=False,
            drop_last=True
            )
        return train_loader, val_loader

    def generate_xy(self, batch):
        latent_classes_batch = batch[0][:, 0].view(-1, 1).to(torch.long)              

        embedder = Embedder(**embed_kwargs)


        coords = batch[0][:, 1:]                            
        encoded_xyz = embedder.embed(coords)

        x = encoded_xyz.to(device)                
        y = batch[1].to(device)     

        return x, y, latent_classes_batch.view(-1), None

    def train(self, train_loader):
        total_loss = 0.0
        iterations = 0.0
        self.model.train()
        for batch in train_loader:

            iterations += 1.0

            self.optimizer_model.zero_grad()

            x, y, latent_codes_indices_batch, latent_codes_batch = self.generate_xy(batch)

            predictions = self.model(x)  
            if self.train_cfg['clamp']:
                predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
            
            loss_value = self.train_cfg['loss_multiplier'] * torch.mean(torch.abs(predictions - y)) 
            loss_value.backward()     

            self.optimizer_model.step()
            total_loss += loss_value.data.cpu().numpy()  

        avg_train_loss = total_loss/iterations
        print(f'Training: loss {avg_train_loss}')
        self.writer.add_scalar('Training loss', avg_train_loss, self.epoch)

        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        iterations = 0.0
        self.model.eval()
        for batch in val_loader:

            iterations += 1.0            

            x, y, _, _ = self.generate_xy(batch)

            predictions = self.model(x)  
            if self.train_cfg['clamp']:
                predictions = torch.clamp(predictions, -train_cfg['clamp_value'], train_cfg['clamp_value'])

            loss_value = self.train_cfg['loss_multiplier'] * torch.mean(torch.abs(predictions - y)) 

            total_loss += loss_value.data.cpu().numpy()   

        avg_val_loss = total_loss/iterations

        print(f'Validation: loss {avg_val_loss}')
        self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
        return avg_val_loss

if __name__=='__main__':
    train_cfg_path = os.path.join(config_files_path , 'train_sdf_ShapeNetAD.yaml')
    if not os.path.exists(train_cfg_path):  
        raise FileNotFoundError(f"Config not found: {train_cfg_path}")
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg)
    trainer()