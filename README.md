
# PASDF: Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation

> 🚧 **This repository is under active development.**  
> The complete codebase, pretrained models, and full documentation will be released soon.

---

## 📘 Overview

**PASDF** is a framework designed for **3D anomaly localization and geometric repair**, leveraging high-quality continuous geometric representations via neural implicit functions.  
It bridges 3D anomaly detection and surface reconstruction by learning a Signed Distance Function (SDF) representation of object geometry.

---

## 🧠 Real3D-AD Example Usage

###  Data Preparation
```bash
python data/extract_sdf_real3d.py
```

###  Training

```bash
python model/train_sdf_ofo_Real3D_AD.py
```

###  Inference / Evaluation

```bash
python Test/AD_test.py --config config_files/test_Real3D_AD.yaml
```

---

##  Requirements

* Python ≥ 3.9
* PyTorch ≥ 1.12
* Open3D ≥ 0.17
* NumPy, YAML, tqdm, scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

⭐ **Stay tuned — the full open-source version of PASDF is coming soon!**

