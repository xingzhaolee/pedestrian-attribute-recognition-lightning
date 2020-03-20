import os
import pickle
import sys

import numpy as np
from scipy.io import loadmat


if __name__ == '__main__':
    root = sys.argv[1]
    dataset_dir = os.path.join("dataset", "rap")
    path_image = os.path.join(root, "images")
    os.makedirs(dataset_dir, exist_ok=True)

    data = loadmat(os.path.join(root, "RAP_annotation.mat"))

    with open(os.path.join(dataset_dir, "attributes.txt"), 'w') as f:
        for idx in range(51):
            f.write(data['RAP_annotation'][0][0][6][idx][0][0] + "\n")

    images, attributes = [], []
    for idx in range(41585):
        images.append(os.path.join(
            path_image, data['RAP_annotation'][0][0][5][idx][0][0]))
        attributes.append(data['RAP_annotation'][0][0][1][idx, :51])
    images = np.array(images)
    attributes = np.array(attributes)

    splits = {}
    splits['train'] = (data['RAP_annotation'][0][0][0][0]
                       [0][0][0][0][0, :] - 1).tolist()
    splits['test'] = (data['RAP_annotation'][0][0][0][0]
                      [0][0][0][1][0, :] - 1).tolist()
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
