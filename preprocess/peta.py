import os
import pickle
import sys

import numpy as np
from scipy.io import loadmat


if __name__ == '__main__':
    root = sys.argv[1]
    dataset_dir = os.path.join("dataset", "peta")
    os.makedirs(dataset_dir, exist_ok=True)

    data = loadmat(os.path.join(root, "PETA.mat"))

    with open(os.path.join(dataset_dir, "attributes.txt"), 'w') as f:
        for idx in range(35):
            f.write(data['peta'][0][0][1][idx, 0][0] + "\n")

    images, attributes = [], []
    for idx in range(19000):
        images.append(os.path.join(root, "images", f"{(idx + 1):05d}.png"))
        attributes.append(data['peta'][0][0][0][idx, 4:4 + 35].tolist())
    images = np.array(images)
    attributes = np.array(attributes)

    splits = {}
    splits['train'] = (data['peta'][0][0][3][0][0][0][0][0][:, 0] - 1).tolist()
    splits['test'] = (data['peta'][0][0][3][0][0][0][0][2][:, 0] - 1).tolist()
    del data

    for split in ['train', 'test']:
        dataset = []
        paths = images[splits[split]]
        labels = attributes[splits[split]]

        if split == 'train':
            ratios = np.mean(labels == 1, axis=0)
            np.save(os.path.join(dataset_dir, "positive_ratios"), ratios)

        for path, label in zip(paths, labels):
            dataset.append((path, label.tolist()))

        with open(os.path.join(dataset_dir, f"{split}.pkl"), 'wb') as f:
            pickle.dump(dataset, f)
