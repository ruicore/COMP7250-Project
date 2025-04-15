## How To Run

### 1. Code Structure
All code are under `project` folder, which contains 4 files:

* `data.py` : Data and its configuration.
* `model.py`: Model and its configuration.
* `train.py`: Training and evaluation.
* `main.py`: Main function to run the project.


### 2. Requirements
Python 3.12 (Python3.13 is not supported)

Install the requirements:
```bash
pip install -r requirements.txt
```
This whole project run approximately **30 hours** on 1 x RTX 3070 (16GB RAM) x16 vCPU 31 GB RAM

### 3. See the Results
After running the code, there will be a folder `./running`, it contains the training logs and the checkpoint files. logs are saved in `./running/logs`.

run `tensorboard --logdir=./running/logs` and open the url in browser to see the logs.

Dut to the training time is relatively long, I have provided an online version of the logs, access it [here](https://wandb.ai/ruihe/COMP7250-Project) (only me can modify the logs, but you can see the results).
Also logs.zip is provided.