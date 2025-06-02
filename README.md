# 🔥 LitResNet Training Framework

This project provides a modular and extensible training pipeline built with **PyTorch Lightning**, designed for experimenting with **ResNet-18** on image classification tasks (e.g., CIFAR-10). It includes support for label smoothing, channels-last memory format, and model compilation using `torch.compile`.

## 🚀 Features

- ✅ PyTorch Lightning integration
- ✅ ResNet-18 backbone
- ✅ Label Smoothing support
- ✅ Channels Last memory optimization
- ✅ Torch Compile modes (`default`, `reduce-overhead`, `max-autotune`)
- ✅ Modular trainer and builder support
- ✅ Metric logging with TensorBoard (train/val accuracy, loss)
- ✅ Hyperparameter tracking via `self.save_hyperparameters()`

## 🏗️ Project Structure

```
.
├── data.py        # LightningDataModule (CIFAR-10 or other)
├── model.py       # LitResNet module + Builder class
├── train.py       # Entry point for training loop
├── main.py        # Optional experiment launcher
```

## 🔧 Example Usage

```bash
python train.py --batch_size 128 --lr 0.01 --num_workers 4
```

Or iterate through model variations:

```python
from model import LitResNetBuilder

for model in LitResNetBuilder():
    trainer.fit(model, datamodule=...)
```

## 🧪 Model Summary (Example)

```
0 | model     | ResNet             | 11.2 M | train
1 | criterion | CrossEntropyLoss   | 0      | train
2 | train_acc | MulticlassAccuracy | 0      | train
3 | val_acc   | MulticlassAccuracy | 0      | train
```

## 📊 Logging

- Training metrics are logged via TensorBoard (`lightning_logs/`)
- Hyperparameters (like `label_smoothing`, `channels_last`, `batch_size`) can be logged in `on_fit_start()` using:

```python
self.logger.log_hyperparams(self.hparams)
```

## ⚙️ Torch Compile Support

The model can optionally be compiled via:

```python
lit.model = torch.compile(lit.model, mode='default')
```

Valid modes include: `off`, `default`, `reduce-overhead`, `max-autotune`.

## 📄 License

This project is licensed under **CC BY-NC 4.0**.

You are free to share and adapt the source code **non-commercially**, with **proper credit**.

## 🎓 Course Info

- **Course:** COMP7250 Machine Learning  
- **Instructor:** HKBU Spring 2024/25
