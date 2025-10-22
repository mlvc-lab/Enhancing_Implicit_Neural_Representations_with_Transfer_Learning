# Enhancing Implicit Neural Representations With Transfer Learning  
**IEEE Access, Accepted September 2025**  
📄 [Paper on IEEE Xplore](https://ieeexplore.ieee.org/document/11152063)

---

## 📘 Overview
This repository contains the official implementation of  
**“Enhancing Implicit Neural Representations With Transfer Learning.”**  
The project explores how transfer learning can significantly improve the convergence and reconstruction performance of Implicit Neural Representations (INRs) across various modalities.

---

## 🧩 Pretrained Models and Datasets
All datasets used for **pretraining** and **inference** can be downloaded from the following Google Drive link:

🔗 [Pretrain & Inference Data – Google Drive](https://drive.google.com/drive/folders/19N7zuj1YFniJ92vzQ60xfRUyKIHqlhbt?usp=drive_link)

---

## ⚙️ Environment Setup

To reproduce the experiments, create and activate a conda environment as follows:

```bash
# Create environment
conda env create -f environment.yaml

# (or manually install using pip)
pip install -r requirements.txt

# Activate environment
conda activate Enhance_INR_with_TL
```

## 🧰 Example Usage
```bash
# Run SIREN experiment on Chest_CT dataset (256×256 resolution)
bash scripts/run_siren.sh --dset Chest_CT --sidelen 256
```

