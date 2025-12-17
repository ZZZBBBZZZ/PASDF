

# 📘 [PASDF: Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](https://arxiv.org/abs/2505.24431)
Bozhong Zheng<sup>1</sup><sup>\*</sup>, Jinye Gan<sup>1</sup><sup>\*</sup>, Xiaohao Xu<sup>2</sup><sup>†</sup>, Xintao Chen<sup>1</sup>,  Wenqiao Li<sup>1</sup>, Xiaonan Huang<sup>2</sup>, Na Ni<sup>1</sup><sup>†</sup>, Yingna Wu<sup>1</sup><sup>†</sup>  

<sup>1</sup>ShanghaiTech University &nbsp;&nbsp; <sup>2</sup>University of Michigan, Ann Arbor  

<sup>\*</sup>Equal contribution. &nbsp;  <sup>†</sup>corresponding author.


---

## 🌍 Overview

**PASDF** is a framework designed for **3D anomaly localization and geometric repair**, leveraging high-quality continuous geometric representations via neural implicit functions.  
It bridges 3D anomaly detection and surface reconstruction by learning a Signed Distance Function (SDF) representation of object geometry. 


## 🧩 1. Installation

```bash
conda create -n PASDF python=3.10
conda activate PASDF
cd path/to/PASDF
bash install.sh PASDF
```


## 📦 2. Download Required Data

### 2.1 Datasets

####  Real3D-AD Dataset

* **Training / Evaluation Data (PCD format)**

  * [Google Drive](https://drive.google.com/file/d/1oM4qjhlIMsQc_wiFIFIVBvuuR8nyk2k0/view?usp=sharing)
  * [Baidu Disk (code: `vrmi`)](https://pan.baidu.com/s/1orQY3DjR6Z0wazMNPysShQ)

* **Raw Data (PLY format)**

  * [Google Drive](https://drive.google.com/file/d/1lHjvyVquuO8-ROOYcnf7O_lliL1Wa36V/view?usp=sharing)
  * [Baidu Disk (code: `vvz1`)](https://pan.baidu.com/s/1BRdJ8oSwrpAPxTOEwUrjdw)

####  Anomaly-ShapeNet (ShapeNetAD) Dataset

* **Training / Evaluation Data (PCD & OBJ format)**

  * [Baidu Disk (code: `case`)](https://pan.baidu.com/s/1Nm50WIU_jx5viozwe59HsQ?pwd=case)
  * [Hugging Face Dataset Page](https://huggingface.co/datasets/Chopper233/Anomaly-ShapeNet)


### 2.2 Preprocessed Templates and Model Weights

Before running inference (`PASDF/Test/AD_test.py`), download the **preprocessed SDF samples** and **pretrained model weights**.

#### 🔗 [Google Drive Folder](https://drive.google.com/drive/folders/1-aeND5tZ_dFp-7BhZHPyZSofDgxRTYRQ?usp=sharing)

This folder contains:

```
data/
results/
```

#### 📁 Directory Structure

Place the contents into your project as follows:

```
PASDF/
├── data/                           ← Template files for both datasets
│   ├── ShapeNetAD/                 ← Template meshes for ShapeNetAD
│   ├── Real3D_AD/                  ← Template meshes for Real3D-AD
│
├── results/                        ← Pretrained weights and preprocessed SDF data
│   ├── ShapeNetAD/
│   │   ├── runs_sdf/               ← Trained model weights (.pt files)
│   │   └── samples_dict_ShapeNetAD.npy   ← Preprocessed SDF samples
│   │
│   ├── Real3D_AD/
│   │   ├── runs_sdf/
│   │   └── samples_dict_Real3D_AD.npy
│
└── Test/
    ├── AD_test.py
    └── infer.py
```


## ⚙️ 3. Usage

### 🧱 3.1 Anomaly_ShapeNet(ShapeNetAD)

> ⚠️ **Note:** If you have already downloaded the preprocessed data and pretrained weights (Section 2.2),
> you can **skip Steps 1 and 2** and go directly to **Step 3: Evaluation**.

#### **Step 1 — Extract SDF Samples**

Configuration: `config_files/extract_sdf_ShapeNetAD.yaml`

```bash
python data/extract_sdf_ShapeNetAD.py
```

#### **Step 2 — Train SDF Model**

Configuration: `config_files/train_sdf_ShapeNetAD.yaml`

```bash
python Train/train_sdf_ShapeNetAD.py
```

#### **Step 3 — Evaluate**

Update the dataset path in `config_files/test_ShapeNetAD.yaml`:

```yaml
dataset:
  name: ShapeNetAD
  dataset_dir: /path/to/Anomaly-ShapeNet-v2/dataset/16384   # ← modify here
```

Run:

```bash
python Test/AD_test.py --config config_files/test_ShapeNetAD.yaml
```





**Implementation Notes**:


* The Chamfer distance in the PAM module was replaced with a **directed variant** for **Real3D-AD**, differing slightly from the paper but yielding better stability and efficiency.

* Adjusting the **`voxel_size`** parameter improves registration accuracy. The optimal per-class values (used for the best results reported in the paper) are provided in `config_files/voxel_sizes.yaml`, with a default of **0.03**.


## 🧩 Reconstruction

### 1. **Modify the YAML Configuration File**

Update the `checkpoint_path` in `config_files/reconstruct_mesh_ShapeNetAD.yaml`:

```yaml
checkpoint_path: results/ShapeNetAD/runs_sdf/ # default path 
mesh_save_dir: results/ShapeNetAD/reconstruct_mesh/
```

### 2. **Run the Reconstruction Script**

After modifying the YAML configuration file, run the reconstruction script using the following command:

```bash
#for Anomaly-ShapeNet dataset
python scripts/reconstruct_mesh.py config_files/reconstruct_mesh_ShapeNetAD.yaml
#for Real3D-AD dataset
python scripts/reconstruct_mesh.py config_files/reconstruct_mesh_Real3D_AD.yaml
```


## Citation

If you find **PASDF** useful in your research, please cite:

```bibtex
@article{zheng2025bridging,
  title={Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation},
  author={Zheng, Bozhong and Gan, Jinye and Xu, Xiaohao and Li, Wenqiao and Huang, Xiaonan and Ni, Na and Wu, Yingna},
  journal={arXiv preprint arXiv:2505.24431},
  year={2025}
}
```


