import torch
import meshplot as mp
import skimage
import numpy as np

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

# mp.offline()

def clamp(x, delta=torch.tensor([[0.1]]).to(device)):
    """Clamp function introduced in the paper DeepSDF.
    This returns a value in range [-delta, delta]. If x is within this range, it returns x, else one of the extremes.

    Args:
        x: prediction, torch tensor (batch_size, 1)
        delta: small value to control the distance from the surface over which we want to mantain metric SDF
    """
    maximum = torch.amax(torch.vstack((x, -delta)))
    minimum = torch.amin(torch.vstack((delta[0], maximum)))
    return minimum


def SDFLoss_multishape(sdf, prediction, x_latent, sigma):
    """Loss function introduced in the paper DeepSDF for multiple shapes."""
    l1 = torch.mean(torch.abs(prediction - sdf))
    l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2))
    loss = l1 + l2
    #print(f'Loss prediction: {l1:.3f}, Loss regulariser: {l2:.3f}')
    return loss, l1, l2


def generate_latent_codes(latent_size, samples_dict):
    latent_codes = torch.tensor([], dtype=torch.float32).reshape(0, latent_size).to(device)
    #dict_latent_codes = dict()
    for i, obj_idx in enumerate(list(samples_dict.keys())):
        #dict_latent_codes[obj_idx] = i
        latent_code = torch.normal(0, 0.01, size = (1, latent_size), dtype=torch.float32).to(device)
        latent_codes = torch.vstack((latent_codes, latent_code))
    latent_codes.requires_grad_(True)
    return latent_codes #, dict_latent_codes


def get_volume_coords(resolution = 50):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
    # Define grid
    grid_values = torch.arange(-1, 1, float(1/resolution)).to(device) # e.g. 50 resolution -> 1/50 
    grid = torch.meshgrid(grid_values, grid_values, grid_values)
    
    grid_size_axis = grid_values.shape[0]

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device)

    return coords, grid_size_axis


def save_meshplot(vertices, faces, path):
    mp.plot(vertices, faces, c=vertices[:, 2], filename=path)


def predict_sdf(latent, coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(coords_batches.device)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
                
            coords = coords.float()
            latent = latent.float()
            latent_tile = torch.tile(latent, (coords.shape[0], 1))

            # print(coords.shape,latent_tile.shape)
            coords_latent = torch.hstack((latent_tile, coords))
            sdf_batch = model(coords_latent)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf


def predict_sdf_ofo(coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(coords_batches.device)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
                
            coords = coords.float()

            sdf_batch = model(coords)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf


def predict_sdf_ofo_PoseEmbedder(coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(coords_batches.device)
    embedder = Embedder(**embed_kwargs)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            coords = embedder.embed(coords)    
            coords = coords.float()

            sdf_batch = model(coords)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf



def extract_mesh(grad_size_axis, sdf):
    # Extract zero-level set with marching cubes
    grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)

    x_max = np.array([1, 1, 1])
    x_min = np.array([-1, -1, -1])
    vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

    return vertices, faces