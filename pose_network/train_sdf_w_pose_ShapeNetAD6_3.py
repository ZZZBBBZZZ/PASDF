import sys
sys.path.append(r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo")
# 过滤出不包含 'liwq' 的路径
sys.path = [path for path in sys.path if 'liwq' not in path]
# sys.path = [path for path in sys.path if 'wza' not in path]
sys.path = [path for path in sys.path if 'zbz' not in path]
import torch
import model.model_sdf as sdf_model
import torch.optim as optim
import data.dataset_sdf as dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import results.runs_sdf as runs
from utils.utils_deepsdf import SDFLoss_multishape
import os
from datetime import datetime
import numpy as np
import time
from utils import utils_deepsdf
import results
from torch.utils.tensorboard import SummaryWriter
import yaml
# import config_files

dataset_path = r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo/data/ShapeNetAD"
results_path = r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results"
results_runs_sdf_path = r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results/ShapeNetAD/runs_sdf"
config_files_path = r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo/config_files"

# Select device. The 'mps' device (macOS M1 architecture) is not supported as it cannot currently handle weith normalisation. 
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
embed_kwargs = {
    'include_input': True,
    'input_dims': 3,  # 3D 坐标
    'max_freq_log2': 4,  # 最大频率 (log2 scale)
    'num_freqs': 10,  # 频率数量
    'log_sampling': True,  # 使用对数采样
    'periodic_fns': [torch.sin, torch.cos],  # 使用 sin 和 cos 编码
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
        self.timestamp_run = datetime.now().strftime('%d_%m_%H%M%S')   # timestamp to use for logging data
        # self.runs_dir = os.path.dirname(str(runs.__file__))               # directory fo all runs
        self.run_dir = os.path.join(results_runs_sdf_path, self.timestamp_run)  # directory for this run
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        # Logging
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.log_path = os.path.join(self.run_dir, 'settings.yaml')
        with open(self.log_path, 'w') as f:
            yaml.dump(self.train_cfg, f)

        # calculate num objects in samples_dictionary, wich is the number of keys
        samples_dict_path = os.path.join(os.path.dirname(results.__file__), train_cfg["dataset"],f'samples_dict_{train_cfg["dataset"]}.npy')
        samples_dict = np.load(samples_dict_path, allow_pickle=True).item()

        #20250217新加
        int2str_dict_path =  os.path.join(os.path.dirname(results.__file__), train_cfg["dataset"],'idx_int2str_dict.npy')  # 替换为你的文件路径
        int2str_dict = np.load(int2str_dict_path, allow_pickle=True).item() 

        # for class_idx in list(samples_dict.keys()):
        #     class_name = int2str_dict[class_idx]
            
        #     save_dir = os.path.join(self.run_dir, class_name)
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        # 通过配置文件或命令行参数指定 class_idx 的范围


        start_idx = 18
        end_idx = 23

        for class_idx in list(samples_dict.keys()):
            # 只选择 class_idx 在 start_idx 到 end_idx 之间的数据
            if start_idx <= class_idx <= end_idx:
                class_name = int2str_dict[class_idx]
                
                save_dir = os.path.join(self.run_dir, class_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                        # instantiate model and optimisers
                self.model = sdf_model.SDFModel_ofo_PoseEmbedder(
                        self.train_cfg['num_layers'], 
                        self.train_cfg['skip_connections'], 
                        inner_dim=self.train_cfg['inner_dim'],
                        PoseEmbedder_size = 60
                    ).float().to(device)

                # define optimisers
                self.optimizer_model = optim.Adam(self.model.parameters(), lr=self.train_cfg['lr_model'], weight_decay=0)
                
                # generate a unique random latent code for each shape
                # self.latent_codes = utils_deepsdf.generate_latent_codes(self.train_cfg['latent_size'], samples_dict)
                # self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)
                
                # Load pretrained weights and optimisers to continue training
                if self.train_cfg['pretrained']:
                    # load pretrained weights
                    self.model.load_state_dict(torch.load(self.train_cfg['pretrain_weights'], map_location=device))

                    # load pretrained optimisers
                    self.optimizer_model.load_state_dict(torch.load(self.train_cfg['pretrain_optim_model'], map_location=device))

                    # retrieve latent codes from results.npy file
                    results_path = self.train_cfg['pretrain_optim_model'].split(os.sep)
                    results_path[-1] = 'results.npy'
                    results_path = os.sep.join(results_path)
                    # load latent codes from results.npy file
                    results_latent_codes = np.load(results_path, allow_pickle=True).item()
                    self.latent_codes = torch.tensor(results_latent_codes['best_latent_codes']).float().to(device)
                    self.optimizer_latent = optim.Adam([self.latent_codes], lr=self.train_cfg['lr_latent'], weight_decay=0)
                    self.optimizer_latent.load_state_dict(torch.load(self.train_cfg['pretrain_optim_latent'], map_location=device))

                if self.train_cfg['lr_scheduler']:
                    self.scheduler_model =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_model, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')
                    # self.scheduler_latent =  torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_latent, mode='min', factor=self.train_cfg['lr_multiplier'], patience=self.train_cfg['patience'], threshold=0.0001, threshold_mode='rel')
                    
                # 在这里添加你的训练逻辑
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
                            # best_latent_codes = self.latent_codes.detach().cpu().numpy()
                            optimizer_model_state = self.optimizer_model.state_dict()
                            # optimizer_latent_state = self.optimizer_latent.state_dict()

                            # np.save(os.path.join(self.run_dir, 'results.npy'), self.results)
                            torch.save(best_weights, os.path.join(save_dir, 'weights.pt'))
                            torch.save(optimizer_model_state, os.path.join(save_dir, 'optimizer_model_state.pt'))
                            # torch.save(optimizer_latent_state, os.path.join(self.run_dir, 'optimizer_latent_state.pt'))
                            # self.results['best_latent_codes'] = best_latent_codes

                        if self.train_cfg['lr_scheduler']:
                            self.scheduler_model.step(avg_val_loss)
                            # self.scheduler_latent.step(avg_val_loss)

                            self.writer.add_scalar('Learning rate (model)', self.scheduler_model._last_lr[0], epoch)
                            # self.writer.add_scalar('Learning rate (latent)', self.scheduler_latent._last_lr[0], epoch)            
                    
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
        """
        Combine latent code and coordinates.
        Return:
            - x: latent codes + coordinates, torch tensor shape (batch_size, latent_size + 3)
            - y: ground truth sdf, shape (batch_size, 1)
            - latent_codes_indices_batch: all latent class indices per sample, shape (batch size, 1).
                                            e.g. [[2], [2], [1], ..] eaning the batch contains the 2nd, 2nd, 1st latent code
            - latent_batch_codes: all latent codes per sample, shape (batch_size, latent_size)
        Return ground truth as y, and the latent codes for this batch.
        """
        latent_classes_batch = batch[0][:, 0].view(-1, 1).to(torch.long)               # shape (batch_size, 1)

        embedder = Embedder(**embed_kwargs)


        # 对坐标进行位置编码


        coords = batch[0][:, 1:]                                  # shape (batch_size, 3)
        encoded_xyz = embedder.embed(coords)
        # latent_codes_batch = self.latent_codes[latent_classes_batch.view(-1)]    # shape (batch_size, 128)

        x = encoded_xyz.to(device)                 # shape (batch_size, 131)
        y = batch[1].to(device)     # (batch_size, 1)

        return x, y, latent_classes_batch.view(-1), None

    def train(self, train_loader):
        total_loss = 0.0
        iterations = 0.0
        self.model.train()
        for batch in train_loader:
            # batch[0]: [class, x, y, z], shape: (batch_size, 4)
            # batch[1]: [sdf], shape: (batch size)
            iterations += 1.0

            self.optimizer_model.zero_grad()
            # self.optimizer_latent.zero_grad()

            x, y, latent_codes_indices_batch, latent_codes_batch = self.generate_xy(batch)

            predictions = self.model(x)  # (batch_size, 1)
            if self.train_cfg['clamp']:
                predictions = torch.clamp(predictions, -self.train_cfg['clamp_value'], self.train_cfg['clamp_value'])
            
            loss_value = self.train_cfg['loss_multiplier'] * torch.mean(torch.abs(predictions - y)) 
            loss_value.backward()     

            # self.optimizer_latent.step()
            self.optimizer_model.step()
            total_loss += loss_value.data.cpu().numpy()  

        avg_train_loss = total_loss/iterations
        print(f'Training: loss {avg_train_loss}')
        self.writer.add_scalar('Training loss', avg_train_loss, self.epoch)

        return avg_train_loss

    def validate(self, val_loader):
        total_loss = 0.0
        # total_loss_rec = 0.0
        # total_loss_latent = 0.0
        iterations = 0.0
        self.model.eval()

        for batch in val_loader:
            # batch[0]: [class, x, y, z], shape: (batch_size, 4)
            # batch[1]: [sdf], shape: (batch size)
            iterations += 1.0            

            x, y, _, latent_codes_batch = self.generate_xy(batch)

            predictions = self.model(x)  # (batch_size, 1)
            if train_cfg['clamp']:
                predictions = torch.clamp(predictions, -train_cfg['clamp_value'], train_cfg['clamp_value'])

            # loss_value, loss_rec, loss_latent = self.train_cfg['loss_multiplier'] * SDFLoss_multishape(y, predictions, latent_codes_batch, self.train_cfg['sigma_regulariser'])    
            
            loss_value = self.train_cfg['loss_multiplier'] * torch.mean(torch.abs(predictions - y)) 

            total_loss += loss_value.data.cpu().numpy()   
            # total_loss_rec += loss_rec.data.cpu().numpy() 
            # total_loss_latent += loss_latent.data.cpu().numpy()

        avg_val_loss = total_loss/iterations
        # avg_loss_rec = total_loss_rec/iterations
        # avg_loss_latent = total_loss_latent/iterations
        print(f'Validation: loss {avg_val_loss}')
        self.writer.add_scalar('Validation loss', avg_val_loss, self.epoch)
        # self.writer.add_scalar('Reconstruction loss', avg_loss_rec, self.epoch)
        # self.writer.add_scalar('Latent code loss', avg_loss_latent, self.epoch)

        return avg_val_loss

if __name__=='__main__':
    train_cfg_path = os.path.join(config_files_path , 'train_sdf_ShapeNetAD.yaml')
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(train_cfg)
    trainer()