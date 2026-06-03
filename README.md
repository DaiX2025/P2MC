# P2MC

This repository contains the official PyTorch implementation of the paper:
**"Tumor Segmentation with Incomplete MRI Modalities via Prototype-driven Progressive Modality Completion"** (Under Review at *Pattern Recognition*).

This repository provides comprehensive instructions for environment setup, data preprocessing, model training, and testing to ensure full reproducibility.

---

## 1. Environment Requirements

To install the required dependencies, it is recommended to use a virtual environment (e.g., Anaconda).

```bash
# Create and activate a conda environment
conda create -n p2mc python=3.8
conda activate p2mc

# Install dependencies
pip install -r requirements.txt
```

```

```

## 2. Pre-trained Model Weights

Due to GitHub's file size limits, our pre-trained model weights are hosted on Google Drive. All links are publicly accessible.

- [brats2018.pth.tar](https://drive.google.com/file/d/1zn7GBtsZFj3u7cf0Eue9hX3kglRf3wRx/view?usp=drive_link)
- [brats2020.pth.tar](https://drive.google.com/file/d/1aIiEbjMdJLtH6GCIqcl4tMEl_JYk5_rv/view?usp=sharing)
- [cervts2024.pth.tar](https://drive.google.com/file/d/1pCq99HmClxL-NqfyDTz5X5jkSokny7so/view?usp=sharing)

**Instructions:** Download the required weights and place them into the `ckpts/` directory in the root folder.

Plaintext

```
P2MC/
├── ckpts/
│   ├── brats2018.pth.tar
│   ├── brats2020.pth.tar
│   └── cervts2024.pth.tar
├── ...
```

## 3. Data Preparation and Preprocessing

1. **Download Datasets:** Please download the official BraTS2018, BraTS2020, and CC2024 datasets from their respective official websites.
2. **Preprocessing:** Run the corresponding preprocessing scripts to format the data for the P2MC network.

Bash

```
# For BraTS 2018 dataset
python preprocess_brats2018.py

# For BraTS 2020 dataset
python preprocess_brats2020.py

# For CC2024 dataset
python preprocess_cc2024.py
```

*Note: Ensure that you update the input/output data paths inside the preprocessing scripts according to your local directories.*

## 4. Configuration (`config.py`)

All hyperparameters, dataset paths, and training settings are centralized in the `config.py` file. Before training or testing, please modify `config.py` to match your local setup:

- `DATASET`: Choose the dataset you are using (e.g., `'BRATS2020'`).
- `TRAIN_DIR` / `VAL_DIR` / `TEST_DIR`: Set these to your preprocessed data paths.
- `MODALITY`: Define the MRI modalities used (e.g., `['falir', 't1c', 't1', 't2']`).
- `BATCH_SIZE` / `LR`: Adjust batch size and learning rate based on your GPU memory.

## 5. Training

Once the data is preprocessed and `config.py` is configured, you can start training the P2MC model from scratch by running:

Bash

```
python main.py --gpu 0
```

*(Change `--gpu 0` to the specific GPU ID you wish to use, or omit it for multi-GPU setups if implemented).*

## 6. Testing and Inference

To evaluate the model or run inference using our pre-trained weights (or your newly trained checkpoints):

1. Open `config.py`.

2. Locate the `RESUME` variable.

3. Replace the empty string with the path to your downloaded/trained checkpoint.

   Python

   ```
   # In config.py
   RESUME = 'ckpts/brats2020.pth.tar'  # Example path
   ```

4. Run the evaluation script:

Bash

```
python main.py --gpu 0 --evaluate
```

## Citation

If you find this repository or our paper useful for your research, please consider citing our work. *(Citation details will be updated upon acceptance).*