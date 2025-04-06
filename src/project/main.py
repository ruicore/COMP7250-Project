import torch
from data import CIFAR10DataModule, DataModuleBuilder
from model import LitResNet, LitResNetBuilder
from train import TrainBuilder


def run_tricks():
    # different model tricks
    model_builder = LitResNetBuilder()
    for model in model_builder:
        data = CIFAR10DataModule()
        trainer = TrainBuilder.get_default(model_builder.trick, model_builder.value)

        print(model.summary())
        print(data.summary())
        trainer.fit(model, datamodule=data)

    # different data tricks
    data_builder = DataModuleBuilder()
    for data in data_builder:
        model = LitResNet()
        trainer = TrainBuilder.get_default(data_builder.trick, data_builder.value)
        print(model.summary())
        print(data.summary())
        trainer.fit(model, datamodule=data)

    # different training tricks
    trainer_builder = TrainBuilder()
    for trainer in trainer_builder:
        model, data = LitResNet(), CIFAR10DataModule(batch_size=256, num_workers=4, pin_memory=True)
        print(model.summary())
        print(data.summary())
        trainer.fit(model, datamodule=data)

    # different torch tricks
    for matmul_precision in ['highest', 'high', 'medium']:
        torch.set_float32_matmul_precision(matmul_precision)
        model, data, trainer = (
            LitResNet(),
            CIFAR10DataModule(batch_size=256, num_workers=4, pin_memory=True),
            TrainBuilder.get_default('MatmulPrecision', matmul_precision),
        )

        print(model.summary())
        print(data.summary())
        trainer.fit(model, datamodule=data)

    torch.set_float32_matmul_precision('high')
    for cudnn_benchmark in [True, False]:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        model, data, trainer = (
            LitResNet(),
            CIFAR10DataModule(batch_size=256, num_workers=4, pin_memory=True),
            TrainBuilder.get_default('CudnnBenchmark', str(cudnn_benchmark)),
        )
        print(model.summary())
        print(data.summary())
        trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    run_tricks()
