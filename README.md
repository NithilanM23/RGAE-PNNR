# RGAE-PNNR: Residual Gated Autoencoder for Anomaly Detection

This repository contains the official PyTorch implementation of the paper **"[Insert Your Paper Title Here]"**.

Our method combines a **Residual Gated Autoencoder (RGAE)** with a **Prototype Nearest Neighbor Retrieval (PNNR)** memory bank to achieve robust unsupervised anomaly detection. It is designed to work effectively on the MVTec AD and VisA datasets.

## 📂 Project Structure

```text
.
├── data/                   # Dataset folder (Download MVTec AD here)
├── train.py                # Main training script
├── test.py                 # Evaluation script (Image & Pixel AUC)
├── heatmap.py              # Visualization script (Generates overlays)
├── requirements.txt        # Python dependencies
├── checkpoints/            # [Auto-generated] Saves trained models (.pth) and banks (.npy)
└── heatmaps/               # [Auto-generated] Saves visualization results
```
🚀 Setup & Installation
1. Clone the Repository:

git clone [https://github.com/NithilanM23/RGAE-PNNR.git](https://github.com/NithilanM23/RGAE-PNNR.git)
cd RGAE-PNNR

2. Install Dependencies
pip install -r requirements.txt

3. Prepare the Dataset
This repository supports the MVTec AD dataset.

Download the dataset from the official website.

Extract it into a data folder so the structure looks like this:

```
RGAE-PNNR/
└── data/
    └── mvtec_ad/
        ├── bottle/
        │   ├── train/
        │   └── test/
        ├── cable/
        └── ...

```
Usage
1. Training
To train the model on a specific category (e.g., candle). This will automatically create a checkpoints/ folder and save the model (.pth) and memory bank (.npy).


python train.py --data_root ./data/mvtec_ad --category candle --epochs 50
Arguments:
```
--data_root: Path to the dataset root folder.

--category: The class name (e.g., bottle, hazelnut).

--epochs: Number of training epochs (default: 5).
```
2. Evaluation
To evaluate the trained model and calculate Image-level AUC and Pixel-level AUC.
```
python test.py --data_root ./data/mvtec_ad --category candle --checkpoint_dir ./checkpoints
```
3. Visualization (Heatmaps)
To generate anomaly heatmaps overlayed on the test images. This will automatically create a heatmaps/ folder and save the images there.
```
python heatmap.py --data_path ./data/mvtec_ad/candle/test/bad --checkpoint_dir ./checkpoin
```
