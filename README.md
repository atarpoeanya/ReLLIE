# ReLLIE: Deep Reinforcement Learning for Customized Low-Light Image Enhancement

This repository contains the official implementation of the ACMMM 2021 paper [ReLLIE: Deep Reinforcement Learning for Customized Low-Light Image Enhancement](link).

## Introduction
To tackle the low-light image enhancement (LLIE) problem, we propose a novel deep reinforcement learning based method, dubbed ReLLIE, for customized low-light enhancement. Specifically,
ReLLIE models LLIE as a markov decision process, i.e., estimating the pixel-wise image-specific curves sequentially and recurrently. Given the reward computed from a set of carefully crafted non-reference loss functions, a lightweight 
network is proposed to estimate the curves for enlightening of a low-light image input. For more details, please refer to our [orginal paper](link).

<p align=center><img width="80%" src="doc/pipeline.pdf"/></p>

## Requirement
* Python 3.5+
* Chainer 5.0+
* Cupy 5.0+
* OpenCV 3.4+

You can install the required libraries by the command `pip install -r requirements.txt`. We checked this code on cuda-10.0 and cudnn-7.3.1.

## Folder sturcture

