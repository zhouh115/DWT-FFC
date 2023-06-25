# Breaking Through the Haze: An Advanced Non-Homogeneous Dehazing Method based on Fast Fourier Convolution and ConvNeXt 
This is the official PyTorch implementation of our dehazing method based on the FFC and ConvNeXt.

Winner award(1st place solution) of NTIRE 2023 HR NonHomogeneous Dehazing Challenge (CVPR Workshop 2023).

See more details in [[Challenge Report](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Ancuti_NTIRE_2023_HR_NonHomogeneous_Dehazing_Challenge_Report_CVPRW_2023_paper.pdf)], [[Paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhou_Breaking_Through_the_Haze_An_Advanced_Non-Homogeneous_Dehazing_Method_Based_CVPRW_2023_paper.pdf)], [[Certificate](https://drive.google.com/file/d/1VyAYLamCxugYw6jWT8SVbg82_z6BkNcZ/view?usp=drive_link)].

# Environment:

CUDA Version: 11.0

Python 3.8

# Dependencies:

torch==1.10.0

torchvision==0.9.0

NVIDIA GPU and CUDA

pytorch_lightning=2.0.0

timm=0.6.12


# Pretrained Model
Download [the pretrained ConvNext model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth) and place it into the folder ./weights. 

# Our saved Model
Download [our saved model for NTIRE 2023 HR Nonhomogeneous Test set](https://drive.google.com/file/d/1HfaVBTkBSU8sScFW30cv9XiOuXAmaByW/view?usp=sharing) and place it into the folder ./weights to reproduce our test result.

Download [our saved model for NTIRE 2023 HR Nonhomogeneous Validation set](https://drive.google.com/file/d/1Sd7QBUYv6xoNwYKncG5HWoSxtb99p6ev/view?usp=sharing) and place it into the folder ./weights to reproduce our validation result.

# How to reproduce our result or recover your hazy image
Download above pretrained and saved models

Prepare NTIRE2023 HR Nongomogeneous Dehazing Chanllenge Validation set and Test set

Run test.py and find results in the folder ./test_result. Please check the test hazy image path (test.py line 12).


# Technical Details

Framework：Pytorch 

Optimizer: AdamW

Learning rate: initial learning rate=0.0001

GPU: RTX2080Ti

Datasets used for training: NTIRE2023 HR Nonhomogeneous Dehazing Chanllenge Train data, NH-HAZE dataset and NH-HAZE2 dataset

Training time: 72h (5000 epoch)

training strategies: auto augmentation

efificiency optimization strategies: multistage learning rate strategy


# Acknowledgement
We thank the authors of [DW-GAN](https://github.com/liuh127/NTIRE-2021-Dehazing-DWGAN), [LaMa](https://advimman.github.io/lama-project/), and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). Part of our code is built on their models.

# Citation
If you find this repository helps, please consider citing:

    
    @InProceedings{Zhou_2023_CVPR,
        author    = {Zhou, Han and Dong, Wei and Liu, Yangyi and Chen, Jun},   
        title     = {Breaking Through the Haze: An Advanced Non-Homogeneous Dehazing Method based on Fast Fourier Convolution and ConvNeXt},  
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},  
        month     = {June},  
        year      = {2023},  
        pages     = {1894-1903}  
    }





