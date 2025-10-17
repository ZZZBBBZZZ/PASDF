import sys
sys.path.append(r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo")
sys.path = [path for path in sys.path if 'liwq' not in path]
# sys.path = [path for path in sys.path if 'wza' not in path]
sys.path = [path for path in sys.path if 'zbz' not in path]
import torch
import os
import model.model_sdf as sdf_model
from utils import utils_deepsdf
import trimesh
from results import runs_sdf
import results
import numpy as np
import config_files
import yaml

"""Extract mesh from an already optimised latent code and network. 
Store the mesh in the same folder where the latent code is located."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
def predict_sdf_ofo_PoseEmbedder_reconstruct(coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)
    embedder = Embedder(**embed_kwargs)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            coords = embedder.embed(coords)    
            coords = coords.float()

            sdf_batch = model(coords)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf

def read_params(cfg):
    """Read the settings from the settings.yaml file. These are the settings used during training."""
    training_settings_path = os.path.join('/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results/ShapeNetAD/runs_sdf' , cfg['folder_sdf'], 'settings.yaml')
    with open(training_settings_path, 'rb') as f:
        training_settings = yaml.load(f, Loader=yaml.FullLoader)

    return training_settings


def reconstruct_object(cfg, latent_code, obj_idx, model, coords_batches, grad_size_axis): 

    sdf = predict_sdf_ofo_PoseEmbedder_reconstruct(coords_batches, model)
    try:
        vertices, faces = utils_deepsdf.extract_mesh(grad_size_axis, sdf)
    except:
        print('Mesh extraction failed')
        return
    
    # save mesh as obj
    mesh_dir = os.path.join('/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results/ShapeNetAD/runs_sdf', cfg['folder_sdf'], 'meshes_training')
    print(mesh_dir)
    if not os.path.exists(mesh_dir):
        os.mkdir(mesh_dir)
    obj_path = os.path.join(mesh_dir, f"mesh_{obj_idx}.obj")
    trimesh.exchange.export.export_mesh(trimesh.Trimesh(vertices, faces), obj_path, file_type='obj')


def main(cfg):
    training_settings = read_params(cfg)

    # Load the model
    obj_ids = cfg["obj_ids"]
    for class_name in obj_ids:
        weights = os.path.join('/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results/ShapeNetAD/runs_sdf', cfg['folder_sdf'],class_name, 'weights.pt')

        model = sdf_model.SDFModel_ofo_PoseEmbedder(
            num_layers=training_settings['num_layers'], 
            skip_connections=training_settings['skip_connections'], 
            inner_dim=training_settings['inner_dim'],
            PoseEmbedder_size=60).float().to(device)

        model.load_state_dict(torch.load(weights, map_location=device))


    
        # Extract mesh obtained with the latent code optimised at inference
        coords, grad_size_axis = utils_deepsdf.get_volume_coords(cfg['resolution'])
        coords = coords.to(device)

        # Split coords into batches because of memory limitations
        coords_batches = torch.split(coords, 100000)
        
        # Load paths
        str2int_path = os.path.join(r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results/ShapeNetAD", 'idx_str2int_dict.npy')
        # results_dict_path = os.path.join('/home/lc/Desktop/wza/gjy/DeepSDF_ofo/results/ShapeNetAD', cfg['folder_sdf'], 'results.npy')
        
        # Load dictionaries
        str2int_dict = np.load(str2int_path, allow_pickle=True).item()
        # results_dict = np.load(results_dict_path, allow_pickle=True).item()

        # for obj_id_path in cfg['obj_ids']:
        #     # Get object index in the results dictionary
        #     obj_idx = str2int_dict[obj_id_path]  # index in collected latent vector
        #     # Get the latent code optimised during training
        #     latent_code = 0

        #     reconstruct_object(cfg, latent_code, obj_idx, model, coords_batches, grad_size_axis)
        # print(len(coords_batches))
        # print(coords_batches[0].shape)
        # coords_batches = torch.tensor(coords_batches).to(device)
        # coords_batches = torch.stack(coords_batches, dim=0)
        model.to(device)
        reconstruct_object(cfg, 0, class_name, model, coords_batches, grad_size_axis)

if __name__ == '__main__':

    cfg_path = os.path.join(r"/home/lc/Desktop/wza/gjy/DeepSDF_ofo/config_files", 'reconstruct_from_latent_ShapeNetAD.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)