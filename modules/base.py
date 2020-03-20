from abc import abstractmethod
from collections import OrderedDict
import logging as log
import os
import pickle

import numpy as np
import pytorch_lightning as pl
import torch
from torchvision.datasets.folder import default_loader
import torchvision.transforms as T


class LightningModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        if self.hparams.dataset == 'pa100k':
            self.n_classes = 26
        elif self.hparams.dataset == 'peta':
            self.n_classes = 35
        elif self.hparams.dataset == 'rap':
            self.n_classes = 51

    @abstractmethod
    def _build_model(self, x):
        """Construct model."""

    @abstractmethod
    def forward(self, x):
        """Forward pass."""

    @abstractmethod
    def criterion(self, outputs, y):
        """Calculate loss."""

    @abstractmethod
    def predict(self, outputs):
        """Prediction."""

    def _evaluate(self, predictions, y, mA_only=False):
        p = np.sum((y == 1), axis=0)
        n = np.sum((y == 0), axis=0)
        tp = np.sum((y == 1) * (predictions == 1), axis=0)
        tn = np.sum((y == 0) * (predictions == 0), axis=0)

        acc_pos = tp / (p + 1e-20)
        acc_neg = tn / (n + 1e-20)
        mA = (acc_pos + acc_neg) / 2

        if mA_only:
            return mA

        p = np.sum((y == 1), axis=1)
        predicted_p = np.sum((predictions == 1), axis=1)

        intersection = np.sum((y * predictions > 0), axis=1)
        union = np.sum((y + predictions > 0), axis=1)

        accuracy = np.mean(intersection / (union + 1e-20))
        precision = np.mean(intersection / (predicted_p + 1e-20))
        recall = np.mean(intersection / (p + 1e-20))
        f1 = 2 * precision * recall / (precision + recall)

        return mA, accuracy, precision, recall, f1

    def validation_step(self, batch, batch_idx):
        # =============================== Batch ===============================
        x, y = batch

        # ============================== Forward ==============================
        outputs = self.forward(x)
        predictions = self.predict(outputs.detach())
        loss = self.criterion(outputs, y)

        # =============================== Logs ===============================
        output = OrderedDict({
            'val_loss': loss,
            'predictions': predictions.cpu().numpy(),
            'labels': y.cpu().numpy()
        })

        return output

    def validation_epoch_end(self, outputs):
        loss = 0
        predictions = []
        labels = []
        for output in outputs:
            loss += output['val_loss']
            predictions.extend(output['predictions'])
            labels.extend(output['labels'])
        loss /= len(outputs)
        predictions = np.array(predictions)
        labels = np.array(labels)

        mA, accuracy, precision, recall, f1 = self._evaluate(
            predictions, labels)
        mA = mA.mean()

        # =============================== Logs ===============================
        tqdm_dict = {
            'val_loss': loss,
            'val_mA': mA,
            'val_acc': accuracy,
            'val_prec': precision,
            'val_recall': recall,
            'val_f1': f1,
        }

        result = {
            'val_loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        return result

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                    self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, self.hparams.learning_rate,
            epochs=self.hparams.epochs,
            steps_per_epoch=len(self.train_dataloader().dataset),
            pct_start=self.hparams.pct_start)

        return [optimizer], [scheduler]

    def __dataloader(self, train):
        transform = []

        if train:
            transform.append(T.RandomHorizontalFlip())
            transform.append(T.Resize((256, 128)))
            transform.append(T.RandomCrop((256, 128), 10))
        else:
            transform.append(T.Resize((256, 128)))

        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))

        dataset = Dataset(self.hparams.dataset,
                          'train' if train else 'test',
                          transform=T.Compose(transform))

        # DistributedSampler will be added by default in Lightning if necessary
        loader = torch.utils.data.DataLoader(
            dataset, self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn
        )

        return loader

    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return self.__dataloader(train=False)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, split, transform):
        super().__init__()
        self.transform = transform

        self.dataset = pickle.load(
            open(os.path.join("dataset", dataset, f"{split}.pkl"), 'rb'))

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = default_loader(img)
        label = torch.tensor(label).float()

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)


def _worker_init_fn(worker_id):
    np.random.seed(0)
