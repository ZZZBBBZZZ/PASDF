import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]
import torch
import open3d as o3d
import numpy as np
import yaml
from matplotlib import cm

from model import SDFModel_ofo_PoseEmbedder                   
from utils import utils_deepsdf, read_params      

class SDFScorer:
    """
    Efficient reusable SDF anomaly scorer.
    Loads the model once and supports repeated inference calls.
    """

    def __init__(self, cfg, cls_name, device="cuda"):
        self.cfg = cfg
        self.cls_name = cls_name
        self.device = device if torch.cuda.is_available() else "cpu"

        # ---- 1. Load infer settings ----
        cfg_infer = cfg["infer"]
        settings_path = cfg_infer.get("settings_path", None)
        if settings_path is None:
            raise ValueError("settings_path must be provided in cfg")

        # ---- 2. Load model hyperparams ----
        training_settings = read_params(settings_path)

        # ---- 3. Build and load model ----
        weights = os.path.join(
            # os.path.dirname(os.path.abspath(__file__)),
            current_dir,
            "..",
            cfg_infer["checkpoint_path"],
            cls_name,
            "weights.pt",
        )
        self.model = SDFModel_ofo_PoseEmbedder(
            num_layers=training_settings["num_layers"],
            skip_connections=training_settings["skip_connections"],
            inner_dim=training_settings["inner_dim"],
            PoseEmbedder_size=60,
        ).float().to(self.device)

        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        # self.model.eval()
        # print(f"[SDFScorer] Loaded model for class: {cls_name} on {self.device}")

    # ---------------------------------------------------
    def infer(self, points):
        """Compute anomaly score for a single point cloud."""
        cfg_infer = self.cfg["infer"]
        # if not isinstance(points, torch.Tensor):
        # ---- 1. Convert to tensor ----
        points_tensor = torch.as_tensor(points, dtype=torch.float32, device=self.device)
    
        # ---- 2. Model forward ----
        with torch.no_grad():
            sdf_pred = utils_deepsdf.predict_sdf_ofo_PoseEmbedder(points_tensor, self.model)

        # ---- 3. Postprocess ----
        sdf_abs = torch.abs(sdf_pred).detach().cpu().numpy()
        sdf_abs[sdf_abs > 100] = 0

        # ---- 4. Compute pixel & object-level scores ----
        anomaly_score_pixel = sdf_abs.flatten().tolist()
        topk = min(cfg_infer["top_k"], len(anomaly_score_pixel))
        topk_mean = np.mean(np.partition(sdf_abs.flatten(), -topk)[-topk:])
        anomaly_score_object = [float(topk_mean)]

        return anomaly_score_pixel, anomaly_score_object
