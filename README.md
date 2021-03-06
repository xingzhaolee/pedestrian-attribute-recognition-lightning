# Pedestrian Attribute Recognition
A repo for pedestrian attribute recognition research using ResNet family as backbone aimed at reproducibility based on PyTorch Lightning. The experiments in this repo can be easily reproduced (seeded) and extended. Experiments are carried out on PyTorch 1.4.0 and torchvision 0.5.0. Results may vary across PyTorch versions even though training is seeded.

## Requirements
- python >= 3.7
- torch >= 1.4.0
- torchvision >= 0.5.0
- pytorch-lightning >= 0.7.1

## How to run
First, preprocess the dataset to generate the respective pickle files.
```bash
python preprocess/pa100k.py ~/Datasets/PA-100K
python preprocess/peta.py ~/Datasets/PETA
python preprocess/rap.py ~/Datasets/RAP
```

Examples on how to train the basic model.
```bash
python train.py -dataset peta -output ~/Logs/PAR/PETA -model Basic
python train.py -dataset rap -output ~/Logs/PAR/RAP -model Basic -gpus 4
python train.py -dataset pa100k -output ~/Logs/PAR/PA100K -model Basic -gpus 4
```

## Results

### RAP
| Model | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Basic | 77.34 | 66.29 | 78.71 | 79.12 | 78.91 |

### PA-100K
| Model | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Basic | 76.76 | 75.69 | 85.63 | 84.92 | 85.27 |

### PETA
| Model | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Basic | 83.98 | 79.41 | 88.07 | 85.57 | 86.80 |
