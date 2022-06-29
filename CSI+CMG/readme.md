# Reproduce CSI+CMG for new SOTA

## Requirements

### Environments

The required packages are as follows:

- python 3.5
- torch 1.2
- torchvision 0.4
- CUDA 10.0
- scikit-learn 0.22

### Datasets

Please download datasets to `./data` and rename the file. Or you may modify the data path in [main.py](main.py).

### Checkpoints

Please download
the [CSI pretrained model](https://drive.google.com/file/d/1rW2-0MJEzPHLb_PAW-LvCivHt-TkDpRO/view?usp=sharing) provided
by [CSI](https://github.com/alinlab/CSI) and save it as `./checkpoint/cifar10_labeled.model`. You can also train your
own model with CSI's code for other settings.

Also, you need to pretrain a CVAE model with CIFAR10 training data according to CMG stage 1, and save the checkpoint
as `./checkpoint/cvae_10class.pkl`.

## Applying CMG and Evaluations

To perform CMG tuning on CSI models and get the final result on CIFAR10 (OOD Detection on different datasets), run this
command:

```
python -m main \
  --device {the available GPU in your cluser, e.g., cuda:0} \
  --params-dict-name './checkpoint/cifar10_labled.model' \
  --params-dict-name2 './checkpoint/cvae_10class.pkl' 
```
