from argparse import ArgumentParser
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


def main(hparams):
    if hparams.model == 'Basic':
        from modules.basic import BasicModel as Module
    model = Module(hparams)

    if hparams.evaluate:
        trainer = pl.Trainer(gpus=1)
        trainer.test(Module.load_from_checkpoint(hparams.evaluate))
        return   

    logger = pl.loggers.TensorBoardLogger(
        save_dir=os.path.join(hparams.output_dir, "logs"), name='')

    try:
        ver = logger.version
    except FileNotFoundError:
        ver = 0

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(hparams.output_dir,
                              "checkpoint",
                              f"version_{ver}",
                              "{val_mA:.2%}_{val_acc:.2%}_{val_prec:.2%}_" +
                              "{val_recall:.2%}_{val_f1:.2%}_{epoch}"),
        monitor='val_f1',
        mode='max'
    )

    trainer = pl.Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[],
        default_save_path=hparams.output_dir,
        gpus=hparams.gpus,
        fast_dev_run=hparams.fast_dev_run,
        max_epochs=hparams.epochs,
        log_save_interval=10,
        distributed_backend=None if hparams.gpus < 2 else 'ddp',
        precision=16 if hparams.use_16_bit else 32,
    )

    trainer.fit(model)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser()

    # ================================= Main =================================
    parser.add_argument(
        '-epochs',
        default=100,
        type=int,
        help="# of epochs (default: 100)"
    )

    parser.add_argument(
        '-evaluate',
        type=str,
        help="checkpoint to be evaluated"
    )

    parser.add_argument(
        '--fast_dev_run',
        action='store_true',
        help="if true uses fast_dev_run for debugging (default: False)"
    )

    parser.add_argument(
        '-gpus',
        type=int,
        default=1,
        help="how many gpus (default: 1)"
    )

    parser.add_argument(
        '--use_16_bit',
        action='store_true',
        help="if true uses Apex for 16-bit training (default: False)"
    )

    # ======================== Dataset and DataLoader ========================
    parser.add_argument(
        '-batch_size',
        default=64,
        type=int,
        help="batch size per GPU (default: 64)"
    )

    parser.add_argument(
        '-dataset',
        type=str,
        choices=['pa100k', 'peta', 'rap'],
        required=True,
        help="dataset"
    )

    parser.add_argument(
        '-output_dir',
        type=str,
        required=True,
        help="output directory"
    )

    parser.add_argument(
        '-num_workers',
        default=8,
        type=int,
        help="# of workers in data loader (default: 8)"
    )

    # ================================= Model =================================
    parser.add_argument(
        '-model',
        type=str,
        choices=['Basic'],
        required=True,
        help="model to be used"
    )

    # ========================= Optimizer & Scheduler =========================
    parser.add_argument(
        '-learning_rate',
        default=0.8,
        type=float,
        help="max learning rate (default: 0.8)"
    )

    parser.add_argument(
        '-momentum',
        default=0.9,
        type=float,
        help="momentum for SGD (default: 0.9)"
    )

    parser.add_argument(
        '-pct_start',
        default=0.1,
        type=float,
        help="The percentage of the cycle (in number of steps) spent "
        "increasing the learning rate. (default: 0.1)"
    )

    parser.add_argument(
        '-weight_decay',
        default=0.0001,
        type=float,
        help="weight decay for SGD (default: 0.0001)"
    )

    # ================================= Args =================================
    tmp = parser.parse_known_args()[0]
    if tmp.model == 'Basic':
        from modules.basic import BasicModel as Module
    parser = Module.add_model_specific_args(parser)

    hyperparams = parser.parse_args()
    hyperparams.output_dir = os.path.join(hyperparams.output_dir,
                                          hyperparams.model,
                                          hyperparams.backbone)

    main(hyperparams)
