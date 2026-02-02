# Project 01 — Deep CNNs on CIFAR-10 🎯

**Short description:**
- This repository collects experiments and notebooks for Project 01: designing and evaluating Deep Convolutional Neural Networks (CNNs) on the **CIFAR-10** dataset.
- Goal: test multiple architectures (VGG, ResNet, Inception, MobileNet, LeNet, etc.), compare performance, save models, and visualize results.

---

## 📦 Repository structure
- `*.ipynb` — Experiment notebooks for different architectures (VGG16, ResNet, Inception, AlexNet, LeNet, ...).
- `models/` — Saved model weights (e.g., `cnn_image_classifier.h5`).
- `images/` — Illustrations and sample images used in notebooks and README.
- `README.md` — This documentation file.

---

## 🔬 Architectures & experiments
Main architectures present in the repository:
- LeNet
- VGG16 (multiple runs / hyperparams)
- ResNet (various versions/runs)
- Inception
- AlexNet
- MobileNet / DenseNet (if included in notebooks)

Each notebook typically contains:
- Data loading and preprocessing for CIFAR-10
- Data augmentation pipeline
- Model construction and configuration
- Training with callbacks (`ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`)
- Evaluation and visualizations (accuracy/loss curves, confusion matrix)

---

## 🚀 Quick start
1. Clone the repository:
```bash
git clone <your-repo-url>
cd Project_01
```

2. Create a virtual environment and install requirements:
```bash
# Using conda
conda create -n project01 python=3.9 -y
conda activate project01
pip install -r requirements.txt
```
(If `requirements.txt` is missing, install the essentials: `tensorflow matplotlib numpy pandas scikit-learn jupyter`)

3. Open a notebook (for example `project_01_VGG16_0.8974999785423279.ipynb`) in Jupyter Lab/Notebook and run cells in order.

4. Or run a training script (if provided):
```bash
python train.py --model vgg16 --epochs 50 --batch-size 128
```

---

## 📈 Results & artifacts
- Best experiment results are recorded in notebook names (e.g., `VGG16_0.8908...ipynb`) and/or saved under the `models/` folder.
- Use the saved models (e.g., `models/cnn_image_classifier.h5`) for quick inference or further fine-tuning.

---

## 🛠️ Suggestions & next steps
- Add a `run_experiments.py` to automate training of multiple architectures and log results (CSV) + checkpoints.
- Add `requirements.txt` and `environment.yml` to make the environment reproducible.
- Create a comparison notebook that aggregates results and produces summary plots of accuracy and loss across experiments.

---

## 📄 License & contact
- License: **MIT** (add a `LICENSE` file if you want to publish under MIT)
- Need help generating `requirements.txt`, creating a training script, or adding an experiments notebook? Tell me which option you want and I will implement it.

---

**Good luck with your project!** ✨

*Updated by GitHub Copilot (Raptor mini (Preview))*
