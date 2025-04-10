from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler


class TrainBuilder:
    trick_name: str = 'None'
    trick_value: str = 'None'

    def __iter__(self):
        for precision in ['32-true', '16-mixed', 'bf16-mixed']:
            yield self.get_default('Precision', precision, precision=precision)

    @staticmethod
    def get_default(trick: str, value: str, **kwargs) -> pl.Trainer:
        root_dir: str = f"running/{datetime.now().strftime('%m%d')}"
        trainer = pl.Trainer(
            max_epochs=100,
            accelerator='gpu',
            devices=1,
            callbacks=[
                ModelCheckpoint(
                    monitor='val_acc',
                    mode='max',
                    save_top_k=1,
                    dirpath=f'{root_dir}/checkpoints/{trick}:{value}',
                    filename='{epoch}-{val_acc:.4f}',
                ),
                LearningRateMonitor(logging_interval='epoch'),
                EarlyStopping(monitor='val_acc', patience=10, mode='max'),
            ],
            logger=[
                TensorBoardLogger(f'{root_dir}/logs', name=trick, version=value),
                CSVLogger(f'{root_dir}/logs_csv', name=trick, version=value),
            ],
            profiler=PyTorchProfiler(
                dirpath=f'{root_dir}/profiler/{trick}',
                filename=value,
                record_shapes=True,
                export_to_chrome=True,
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{root_dir}/profiler/{trick}:{value}'),
            ),
            log_every_n_steps=20,
            deterministic=False,
            enable_progress_bar=False,
            **kwargs,
        )
        setattr(trainer, 'trick_name', trick)
        setattr(trainer, 'trick_value', value)
        return trainer
