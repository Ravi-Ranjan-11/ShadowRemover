
# ShadowRemover
This is the official implementation of Shadow Removal technique using PyTorch.


## Introduction
To trackle image shadow removal problem, we propose a novel transformer-based method, dubbed ShadowFormer, for exploiting non-shadow
regions to help shadow region restoration. A multi-scale channel attention framework is employed to hierarchically
capture the global information. Based on that, we propose a Shadow-Interaction Module (SIM) with Shadow-Interaction Attention (SIA) in the bottleneck stage to effectively model the context correlation between shadow and non-shadow regions. 


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
* SRD [[Training]](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view)[[Testing]](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view)


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
