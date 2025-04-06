import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler


class TrainBuilder:
    trick: str = ''
    value: str = ''

    def __iter__(self):
        for precision in ['32-true', '16-mixed', 'bf16-mixed']:
            yield self.get_default('Precision', precision, precision=precision)

    @staticmethod
    def get_default(trick: str, value: str, **kwargs) -> pl.Trainer:
        return pl.Trainer(
            max_epochs=100,
            accelerator='gpu',
            devices=1,
            callbacks=[
                ModelCheckpoint(monitor='val_acc', mode='max', save_top_k=1, filename='{epoch}-{val_acc:.4f}'),
                LearningRateMonitor(logging_interval='epoch'),
            ],
            logger=[
                TensorBoardLogger('logs', name=trick, version=value),
                CSVLogger('logs_csv', name=trick, version=value),
                EarlyStopping(
                    monitor='val_acc',
                    patience=10,
                    mode='max',
                ),
            ],
            profiler=PyTorchProfiler(
                dirpath=f'profiler/{trick}',
                filename=value,
                record_shapes=True,
                export_to_chrome=True,
                use_cuda=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'profiler/{trick}:{value}'),
            ),
            log_every_n_steps=20,
            deterministic=False,
            **kwargs,
        )
