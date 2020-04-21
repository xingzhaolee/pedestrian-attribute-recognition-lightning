from argparse import ArgumentParser
from collections import OrderedDict
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image

from .basic import BasicModel
from modules import backbone


class VisualAttentionConsistency(BasicModel):

    def forward(self, x):       
        feature = x = self.backbone(x)

        x = F.adaptive_avg_pool2d(x, 1)

        x = torch.flatten(x, 1)
        x = F.dropout(x, self.hparams.dropout, training=self.training)
        x = self.classifier(x)
        
        if self.training:  # Generate heatmap during training
            fc_weights = self.classifier.weight.data
            fc_weights = fc_weights.view(1, self.n_classes, feature.shape[1], 1, 1)
            feature = feature.unsqueeze(1)
            heatmap = feature * fc_weights
            heatmap = heatmap.sum(2)

            return x, heatmap
        
        return x

    def criterion(self, outputs, y, heatmap=None, heatmap_f=None):
        if not self.hparams.disable_weighted_loss:
            weight = torch.where(
                y.cpu() == 1, self.weight['positive'], self.weight['negative'])
            if self.on_gpu:
                weight = weight.cuda(outputs.device.index)
        else:
            weight = None
        loss = F.binary_cross_entropy_with_logits(outputs, y, weight)
        
        if self.training:  # Heatmap loss
            heatmap_f = heatmap_f.flip(-1)  # Flip back to original
            loss += F.mse_loss(heatmap, heatmap_f)
        
        return loss

    def training_step(self, batch, batch_idx):
        # =============================== Batch ===============================
        x, y = batch
        y_c = torch.cat([y, y])
        
        # ============================== Forward ==============================
        outputs, heatmap = self.forward(x)
        outputs_f, heatmap_f = self.forward(x.flip(-1))  # Horizontal flipped input
        outputs_c = torch.cat([outputs, outputs_f])
        loss = self.criterion(outputs_c, y_c, heatmap, heatmap_f)
        
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
