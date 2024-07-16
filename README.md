
# Code for SAME: Sample-reconstruction Against Model Extraction Attacks

## Requirements
1. PyTorch >= 1.13.1
2. torchvision
3. numpy
4. sklearn
5. scipy

## Evaluate SAME defense on MNIST dataset

```shell
python evaluate_main.py --exp_id main-mnist --model conv3 --dataset mnist --dataset_ood kmnist --proxyset emnist_digits --budget 4000 --attacker knockoff --defenders 'SAME' >> evaluate_main_mnist.log
```

## Evaluate SAME defense on MNIST dataset

```shell
python evaluate_main.py --exp_id main-cifar10 --model res18 --dataset cifar10 --dataset_ood imagenet_tiny --proxyset cifar100 --budget 10000 --attacker knockoff --defenders 'SAME' >> evaluate_main_cifar10.log
```