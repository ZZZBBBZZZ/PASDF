# -*- coding: utf-8 -*-
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../PASDF")))
sys.path = [path for path in sys.path if 'liwq' not in path and 'zbz' not in path]

# from utils.io_utils import set_seed
# from utils.metrics import auroc
# from utils import infer
import argparse, os, yaml, numpy as np
from torch.utils.data import DataLoader
import torch
from dataset import Dataset_Real3D_AD_test
from dataset import Dataset_ShapeNetAD_test
from utils import set_seed
from sklearn.metrics import roc_auc_score
from infer import SDFScorer
from tqdm import tqdm
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_dataset(cfg: dict, cls_name: str):
    ds_cfg = cfg["dataset"]
    name = ds_cfg["name"]
    if name == "Real3D_AD":
        return Dataset_Real3D_AD_test(
            dataset_dir=ds_cfg["dataset_dir"],
            cls_name=cls_name,
            num_points=ds_cfg.get("num_points", 0),
            normalize=ds_cfg.get("normalize", True),
            scale_factor=ds_cfg.get("scale_factor", 17.744022369384766),
            template_path = ds_cfg.get("template_path", None)
        )
    elif name == "ShapeNetAD":
        return Dataset_Real3D_AD_test(
            dataset_dir=ds_cfg["dataset_dir"],
            cls_name=ds_cfg["cls_name"],
            num_points=ds_cfg.get("num_points", 0),
            normalize=ds_cfg.get("normalize", False),
            scale_factor=ds_cfg.get("scale_factor", 1.0),
            template_path = ds_cfg.get("template_path", None),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

def evaluate(loader, cfg, cls_name):
    model = SDFScorer(cfg, cls_name, device=DEVICE)
    pixel_scores, pixel_labels = [], []
    obj_scores, obj_labels = [], []
    for batch in tqdm(loader, desc=f"Evaluating [{cls_name}]", ncols=100):
        pts, mask, label, _ = batch["points"], batch["mask"], batch["label"], batch["path"]
        pts = pts
        pixel_score, object_score = model.infer(pts)
    
        pixel_scores.extend(pixel_scores)
        pixel_labels.extend(mask[0].flatten())
        obj_scores.append(object_score)
        obj_labels.append(label[0])
    pix_auc = roc_auc_score(pixel_labels, pixel_scores)
    obj_auc = roc_auc_score(obj_labels, obj_scores)
    return pix_auc, obj_auc

def main(args):

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    set_seed(cfg["seed"])
    for cls_name in cfg['dataset']['cls_name']:
            
        ds = build_dataset(cfg, cls_name)
        loader = DataLoader(ds, batch_size=cfg['infer']['batch_size'], shuffle=cfg['infer']['shuffle'], num_workers=cfg['infer']['batch_size'])
          
        pix_auc, obj_auc = evaluate(loader, cfg, cls_name)
        print(f"---{cls_name}-- AUROC Pixel: {pix_auc}, AUROC Object: {obj_auc}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    main(args)
