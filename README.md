
# ShadowRemover
This is the official implementation of Shadow Removal technique using PyTorch.


## Introduction
To trackle image shadow removal problem, we propose a novel transformer-based method, dubbed ShadowFormer, for exploiting non-shadow
regions to help shadow region restoration. A multi-scale channel attention framework is employed to hierarchically
capture the global information. Based on that, we propose a Shadow-Interaction Module (SIM) with Shadow-Interaction Attention (SIA) in the bottleneck stage to effectively model the context correlation between shadow and non-shadow regions. 

We have trained the model with different activation functions to check for better results:
1. GELU
2. Swish (SiLU)
3. LeakyRelu


## Requirement
* Python 3.7
* Pytorch 1.7
* CUDA 11.1
```bash
pip install -r requirements.txt
```

## Datasets
* ISTD [[link]](https://github.com/DeepInsight-PCALab/ST-CGAN)  
* ISTD+ [[link]](https://github.com/cvlab-stonybrook/SID)


## Pretrained models
[ISTD](https://drive.google.com/file/d/1bHbkHxY5D5905BMw2jzvkzgXsFPKzSq4/view?usp=share_link) | [ISTD+](https://drive.google.com/file/d/10pBsJenoWGriZ9kjWOcE4l4Kzg-F1TFd/view?usp=share_link) | [SRD]()

Please download the corresponding pretrained model and modify the `weights` in `test.py`.

## Test
You can directly test the performance of the pre-trained model as follows
1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the `test.py` 
```python
input_dir # shadow image input path -- Line 27
weights # pretrained model path -- Line 31
```
2. Test the model
```python
python test.py --save_images
```
You can check the output in `./results`.


## Train

1. Download datasets and set the following structure
```
|-- ISTD_Dataset
    |-- train
        |-- train_A # shadow image
        |-- train_B # shadow mask
        |-- train_C # shadow-free GT
    |-- test
        |-- test_A # shadow image
        |-- test_B # shadow mask
        |-- test_C # shadow-free GT
```
2. You need to modify the following terms in `option.py`
```python
train_dir  # training set path
val_dir   # testing set path
gpu: 0 # Our model can be trained using a single RTX A5000 GPU. You can also train the model using multiple GPUs by adding more GPU ids in it.
```
3. Train the network

To train the model using different types of activation function copy the content of that model in the model.py:
1. For GELU activation function copy the content of model_gelu.py to model.py
2. For Swish activation function copy the content of model_swish.py to model.py
3. For LeakyRelu activation function copy the content of model_leakyRelu.py to model.py

If you want to train the network on 256X256 images:
```python
python train.py --warmup --win_size 8 --train_ps 256
```
or you want to train on original resolution, e.g., 480X640 for ISTD:
```python
python train.py --warmup --win_size 10 --train_ps 320
```



