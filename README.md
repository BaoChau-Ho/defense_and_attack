# The purpose of this repository
This repository contains codes for the adversarial attacks and defenses in this paper: [blank link]()
# Adversarial Attacks
The adversarial attacks that the paper focus on are as following:
|Name | Paper|
|----|----|
| FGSM (Linf)| Explaining and harnessing adversarial examples ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572)) |
| BIM (Linf)| Adversarial Examples in the Physical World ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533)) |
| PGD (L2) | Towards Deep Learning Models Resistant to Adversarial Attacks ([Mardry et al., 2017](https://arxiv.org/abs/1706.06083))|
| CW (L2) | Towards Evaluating the Robustness of Neural Networks ([Carlini et al., 2016](https://arxiv.org/abs/1608.04644)) |

We generate attacks from four different image sets: CIFAR10, CIFAR100, MNIST and IMAGENET; each set uses 5 different backbone models: Resnet 50, Resnet 100, MobileNet, DenseNet, AlexNet and Inception v3 Net
The code used to generate the attacks and their top 1 and top 5 accuracies is adversarial_attacks_get_results.py

Usage: We index the datasets and models used as backbone in the code, and so we shall use it as inputs of the code
* `--index_dataset` is the index of the dataset (adhering to the way we index it in the code)
* `--index_model` is the index of the model used as backbone in respect to the dataset (adhering to the way we index it in the code)
* `--father_directory` is the directory that you store the dataset, the model and the generated results (both accuracies and adversarial images)
* `--number_of_imgs` is the number of adversarial images you wish to save
* Note:
   - Datasets and Models should be prepared beforehand: Most datasets can be downloaded via pytorch's Dataset, except for IMGNET. Models from pytorch are trained on IMGNET, while the code requires them to be trained on other datasets as well.
   - The hyperparameters for the adversarial methods, the batch size, etc can be modified according to your needs




