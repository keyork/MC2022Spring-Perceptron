# MC2022Spring-Perceptron

Homework 1 of MC2022Spring(Media and Cognition 2022 Spring)

## Introduction

- A simple demo of a perceptron.

- The dataset is a set of linearizable samples, there are 50 positive and negative samples each.

- Optimization was performed using the stochastic gradient descent method.

## Requirements

```
colorlog==6.6.0
matplotlib==3.5.1
numpy==1.21.5
```

## Run Demo

1. Make sure you have the folder: `'./img/'`
2. Enjoy it!

```
python train.py
```

## Result

- Dataset

  ![raw_data_plot](./resources/raw_data_plot.png)

- Result of model A

  ![final_modelA](./resources/final_modelA.png)

- Optim process of model A

  ![result_modelA](./resources/result_modelA.png)

- Loss of model A

  ![loss_modelA](./resources/loss_modelA.png)

- Result of model B

  ![final_modelB](./resources/final_modelB.png)

- Optim process of model B

  ![result_modelB](./resources/result_modelB.png)

- Loss of model B

  ![loss_modelB](./resources/loss_modelB.png)

## Output in Terminal

![image-20220317004044989](./resources/terminal.png)