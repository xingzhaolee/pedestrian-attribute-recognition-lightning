import os
import pickle
import sys

import numpy as np
from scipy.io import loadmat


if __name__ == '__main__':
    root = sys.argv[1]
    dataset_dir = os.path.join("dataset", "pa100k")
    os.makedirs(dataset_dir, exist_ok=True)

    data = loadmat(os.path.join(root, "annotation.mat"))

    with open(os.path.join(dataset_dir, "attributes.txt"), 'w') as f:
        for idx in range(26):
            f.write(data['attributes'][idx][0][0] + "\n")

    images, attributes = [], []
    for idx in range(80000):
        p = data['train_images_name'][idx][0][0]
        images.append(os.path.join(root, "data",
                                   "release_data", "release_data", p))
        attributes.append(data['train_label'][idx, :].tolist())
    for idx in range(10000):
        p = data['test_images_name'][idx][0][0]
        images += [os.path.join(root, "data",
                                "release_data", "release_data", p)]
        attributes += [data['test_label'][idx, :].tolist()]
    images = np.array(images)
    attributes = np.array(attributes)

    splits = {}
    splits['train'] = range(80000 - 1)
    splits['test'] = [i + 80000 for i in range(10000 - 1)]
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
