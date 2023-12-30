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

These weights are the checkpoints that perform best for NTIRE 23 dehazing challenge ofiicial validation set and test set.
# How to reproduce our result or recover your hazy image
Download above pretrained and saved models

Prepare NTIRE2023 HR Nongomogeneous Dehazing Chanllenge Validation set and Test set

Run test.py and find results in the folder ./test_result. Please check the test hazy image path (test.py line 12).


# Technical Details for NTIRE 23 Non-homogeneous Dehazing Challenge

Framework：Pytorch 

Optimizer: AdamW

Learning rate: initial learning rate=0.0001

GPU: RTX2080Ti

Datasets used for training: NTIRE2023 HR Nonhomogeneous Dehazing Chanllenge Train data, NH-HAZE dataset and NH-HAZE2 dataset

Training time: 72h (5000 epoch)

training strategies: auto augmentation

efificiency optimization strategies: multistage learning rate strategy

# More Information about our model and paper
Datasets can be found below:

[Reside](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=3D0), [NH-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/), [NH-HAZE2](https://drive.google.com/drive/folders/1YLYgUm7ttjdqfvuFiYCm-33Xe3J3Jnir?usp=sharing), [HD-NH-HAZE](https://drive.google.com/drive/folders/1_8zZ6-MEMubZb2NC-6UPKezqQDPWt2vL?usp=sharing), and our [Combined Dataset](https://drive.google.com/drive/folders/1cBzxP51d56Y9W1BlGbEdePUYZmRVTqBS?usp=sharing).

If you want to train with your data, you can use the train.py in [DW-GAN](https://github.com/liuh127/NTIRE-2021-Dehazing-DWGAN), as we adopt similar training stratety with DWGAN.

We are sorry that we didn't name our model in our paper, but we are glad you can use DWT-FFC to represent our method if you want to compare with our model.



# Acknowledgement
We thank the authors of [DW-GAN](https://github.com/liuh127/NTIRE-2021-Dehazing-DWGAN), [LaMa](https://advimman.github.io/lama-project/), and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). Part of our code is built on their models.

# Citation
If you find this repository helps, please consider citing:

    
    @InProceedings{Zhou_2023_CVPRW,
        author    = {Zhou, Han and Dong, Wei and Liu, Yangyi and Chen, Jun},   
        title     = {Breaking Through the Haze: An Advanced Non-Homogeneous Dehazing Method based on Fast Fourier Convolution and ConvNeXt},  
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},  
        month     = {June},  
        year      = {2023},  
        pages     = {1894-1903}  
    }





