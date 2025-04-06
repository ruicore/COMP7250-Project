import time

import torch
from data import CIFAR10DataModule, DataModuleBuilder
from model import LitResNet, LitResNetBuilder
from train import TrainBuilder


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ Function '{func.__name__}' executed in {(end_time - start_time):.2f} seconds.")
        return result

    return wrapper


@timer
def run_tricks():
    # different model tricks
    model_builder = LitResNetBuilder()
    for model in model_builder:
        data = CIFAR10DataModule()
        trainer = TrainBuilder.get_default(model_builder.trick, model_builder.value)

        print('📦 Model Config:', model.summary())
        print('📊 Data Config:', data.summary())
        print('🚀 Training started...\n')

        trainer.fit(model, datamodule=data)
        print('✅ Training complete\n')
        trainer.test(model, datamodule=data)

    # different data tricks
    data_builder = DataModuleBuilder()
    for data in data_builder:
        model = LitResNet()
        trainer = TrainBuilder.get_default(data_builder.trick, data_builder.value)
        print('📦 Model Config:', model.summary())
        print('📊 Data Config:', data.summary())
        print('🚀 Training started...\n')
        trainer.fit(model, datamodule=data)
        trainer.test(model, datamodule=data)

    # different training tricks
    trainer_builder = TrainBuilder()
    for trainer in trainer_builder:
        model, data = LitResNet(), CIFAR10DataModule()
        print('📦 Model Config:', model.summary())
        print('📊 Data Config:', data.summary())
        print('🚀 Training started...\n')
        trainer.fit(model, datamodule=data)
        trainer.test(model, datamodule=data)

    # different torch tricks
    for matmul_precision in ['highest', 'high', 'medium']:
        torch.set_float32_matmul_precision(matmul_precision)
        model, data, trainer = (
            LitResNet(),
            CIFAR10DataModule(),
            TrainBuilder.get_default('MatmulPrecision', matmul_precision),
        )

        print('📦 Model Config:', model.summary())
        print('📊 Data Config:', data.summary())
        print('🚀 Training started...\n')

        trainer.fit(model, datamodule=data)
        print('✅ Training complete\n')
        trainer.test(model, datamodule=data)

    torch.set_float32_matmul_precision('high')
    for cudnn_benchmark in [True, False]:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        model, data, trainer = (
            LitResNet(),
            CIFAR10DataModule(),
            TrainBuilder.get_default('CudnnBenchmark', str(cudnn_benchmark)),
        )
        print('📦 Model Config:', model.summary())
        print('📊 Data Config:', data.summary())
        print('🚀 Training started...\n')
        trainer.fit(model, datamodule=data)
        trainer.test(model, datamodule=data)


if __name__ == '__main__':
    run_tricks()
