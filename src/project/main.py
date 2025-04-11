import torch
from data import CIFAR10DataModule, DataModuleBuilder
from model import LitResNet, LitResNetBuilder
from train import TrainBuilder


def run_tricks():
    # data tricks
    for data in DataModuleBuilder():
        model = LitResNet(
            batch_size=data.batch_size,
            num_workers=data.num_workers,
            pin_memory=data.pin_memory,
            trick_name=data.trick_name,
            trick_value=data.trick_value,
        )
        trainer = TrainBuilder.get_default(data.trick_name, data.trick_value)
        print(model.summary(), data.summary(), sep='\n')
        trainer.fit(model, datamodule=data)

    # model tricks
    for model in LitResNetBuilder():
        data = CIFAR10DataModule(trick_name=model.trick_name, trick_value=model.trick_value)
        trainer = TrainBuilder.get_default(model.trick_name, model.trick_value)
        print(model.summary(), data.summary(), sep='\n')
        trainer.fit(model, datamodule=data)

    # training tricks
    for trainer in TrainBuilder():
        print('Trainer:', trainer.trick_name, trainer.trick_value)
        trainer.fit(LitResNet(), datamodule=CIFAR10DataModule())

    for matmul_precision in ['highest', 'high', 'medium']:
        torch.set_float32_matmul_precision(matmul_precision)
        model, data, trainer = (
            LitResNet(),
            CIFAR10DataModule(),
            TrainBuilder.get_default('MatmulPrecision', matmul_precision),
        )
        print('Matmul Precision:', matmul_precision)
        trainer.fit(model, datamodule=data)

    torch.set_float32_matmul_precision('high')
    for cudnn_benchmark in [True, False]:
        torch.backends.cudnn.benchmark = cudnn_benchmark
        model, data, trainer = (
            LitResNet(),
            CIFAR10DataModule(),
            TrainBuilder.get_default('CudnnBenchmark', str(cudnn_benchmark)),
        )
        print('Cudnn Benchmark:', cudnn_benchmark)
        trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    run_tricks()
