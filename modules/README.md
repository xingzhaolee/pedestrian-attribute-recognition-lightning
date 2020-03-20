## Modules

### Basic
Plain model based on ResNet family.

## Results

### Basic

#### RAP
| Model | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| + LR tuning | 77.34 | 66.29 | 78.71 | 79.12 | 78.91 |
| + Weighted Sigmoid Cross Entropy | 75.52 | 65.87 | 78.67 | 78.56 | 78.61 |
| Basic | 70.81 | 65.60 | 82.38 | 74.64 | 78.32 |

#### PA-100K
| Model | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| + LR tuning (based on RAP) | 76.76 | 75.69 | 85.63 | 84.92 | 85.27 |
| + Weighted Sigmoid Cross Entropy | 76.28 | 75.57 | 86.26 | 84.12 | 85.18 |
| Basic | 73.96 | 75.55 | 88.46 | 82.02 | 85.12 |

#### PETA
| Model | mA | Accuracy | Precision | Recall | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| + LR tuning (based on RAP) | 83.98 | 79.41 | 88.07 | 85.57 | 86.80 |
| + Weighted Sigmoid Cross Entropy | 82.30 | 76.80 | 86.66 | 83.45 | 85.03 |
| Basic | 80.90 | 75.96 | 86.63 | 82.38 | 84.45 |
