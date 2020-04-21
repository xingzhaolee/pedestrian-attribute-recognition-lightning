from argparse import ArgumentParser
from collections import OrderedDict
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import LightningModule
from modules import backbone


class BasicModel(LightningModule):

    def __init__(self, hparams):
        super().__init__(hparams)

        if not self.hparams.disable_weighted_loss:
            positive_ratios = np.load(
                os.path.join("dataset", self.hparams.dataset,
                             "positive_ratios.npy"))
            self.weight = {}
            self.weight['positive'] = torch.from_numpy(
                np.exp(1. - positive_ratios)).float()
            self.weight['negative'] = torch.from_numpy(
                np.exp(positive_ratios)).float()

        self._build_model()

    def _build_model(self):
        self.backbone, feature_size = backbone.resnet(
            self.hparams.backbone,
            pretrained=not self.hparams.disable_pretrained_imagenet)

        self.classifier = nn.Linear(feature_size, self.n_classes)

    def forward(self, x):
        x = self.backbone(x)

        x = F.adaptive_avg_pool2d(x, 1)

        x = torch.flatten(x, 1)
        x = F.dropout(x, self.hparams.dropout, training=self.training)
        x = self.classifier(x)

        return x

    def criterion(self, outputs, y):
        if not self.hparams.disable_weighted_loss:
            weight = torch.where(
                y.cpu() == 1, self.weight['positive'], self.weight['negative'])
            if self.on_gpu:
                weight = weight.cuda(outputs.device.index)
        else:
            weight = None
        loss = F.binary_cross_entropy_with_logits(outputs, y, weight)
        return loss

    def predict(self, outputs):
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        return predictions

    def training_step(self, batch, batch_idx):
        # =============================== Batch ===============================
        x, y = batch

        # ============================== Forward ==============================
        outputs = self.forward(x)
        loss = self.criterion(outputs, y)

        # ============================== Metrics ==============================
        predictions = self.predict(outputs.detach())
        mA = self._evaluate(predictions.cpu().numpy(), y.cpu().numpy(),
                            mA_only=True).mean()

        # =============================== Logs ===============================
        tqdm_dict = {'train_mA': mA}

        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    @staticmethod
    def add_model_specific_args(parent):
        parser = ArgumentParser(parents=[parent], add_help=False)

        parser.add_argument(
            '-backbone',
            default='resnet50',
            type=str,
            choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                     'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                     'wide_resnet50_2', 'wide_resnet101_2'],
        )

        parser.add_argument(
            '-dropout',
            default=0.,
            type=float,
        )

        parser.add_argument(
            '--disable_pretrained_imagenet',
            action='store_true',
        )

        parser.add_argument(
            '--disable_weighted_loss',
            action='store_true',
        )

        return parser
