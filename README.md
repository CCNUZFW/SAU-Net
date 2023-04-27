# SAU-Net: Saliency-based Adaptive Unfolding Network for Interpretable High Quality Image Compressed Remote Sensing

## About 
This is an implementation of the SAU-Net model referring to the following paper: SAU-Net: Saliency-based Adaptive Unfolding Network for Interpretable High Quality Image Compressed Remote Sensing

## Contributors
1. Shiyan Xia : 102110268@hbut.edu.cn
2. Zhifeng Wang : zfwang@ccnu.edu.cn</br>

School of Electrical and Electronic Engineering, Hubei University of Technology, Wuhan 430068, China

## Environment Requirement
python == 3.7.12</br>

torch == 1.11.0</br>

numpy == 1.21.5</br>

## Datasets
1. The training set in this paper is the brightness components of 25600 images extracted 428 from T91 and Train400 with a size of 128×128.The testing sets in this paper are Set11 and BSD68.

2. The preprocessed training set file [Training_Data_size128_CS_T91andTrain400.mat] is put into `./data`. If not, please download it from (https://).

3. The testing sets are placed in `./data/Set11` and `./data/BSD68`. 

## Train 
1. Run the `train.py` to train.

2. The log and model files will be in `./log` and `./model`, respectively.

## Test
1. Download pre-trained model file [net_params_320.pkl](https://) and put it into `./model/layer_13_block_32`.

2. Select the testing set,and then run `test.py`.

	```shell
	python test.py --testset_name=Set11
	python test.py --testset_name=BSD68
	```
	
3. The reconstruction images will be in `./test_out`.
   The reconstruction images named as **imagename_SAUNet_ratio_num1_epoch_num2_PSNR_num3_SSIM_num4**,
   where **num1** is the CS ratio, **num2** is the number of the iteration, **num3** is the PSNR of the image and  **num4** is the SSIM of the image.

