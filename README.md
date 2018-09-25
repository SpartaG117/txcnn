# 系统版本

ubuntu16.04

# Python版本

python3.6

# Python package

pytorch 0.4.1

torchvision 0.2.1 

tqdm 4.23.4

opencv3 

Pillow 5.1.0

# 神经网络模型

见Model.py

# 训练策略

Adam初始学习率1e-4

# 外部的预训练模型

resnet152，torchvison.models中调用自动下载，imagenet预训练

# 请确保data目录结构如下
    |--data
        |-- DatasetA
                |-- train
                |-- test
                |-- train.txt
                |-- test.txt
                |-- submit.txt
        |-- DatasetB
                |-- train
                |-- test
                |-- train.txt
                |-- test.txt
