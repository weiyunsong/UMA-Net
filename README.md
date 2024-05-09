# A lightweight attention-based network for image dehazing 
In current convolution-based image dehazing networks, a common approach to improve network performance is to increase the depth and width of the convolutions. This strategy significantly increases the network complexity and computational cost. Therefore, this paper proposes a novel Multiscale Adaptive (MA) image dehazing module, which consists of the Multi-Scale Kernel (MSK) combination module and the Adaptive Channel and Spatial Selection (ASCS) module, along with a lightweight Channel Attention Guided Fusion (CAGF) module. The MSK effectively enlarges the receptive field without introducing additional parameters or computational cost. The ASCS combines standard convolutions with dilated convolutions to further focus on spatial and channel information in the forward propagation network. Using the MA and CAGF modules, a U-shaped Multiscale Adaptive Network (UMA-Net) is constructed to efficiently restore high-quality haze-free images from hazy images. Extensive experiments demonstrate the effectiveness of the proposed modules, which achieve state-of-the-art performance on the Reside SOTS dataset with only 0.816M parameters and 8.794G FLOPs.
![image](https://github.com/weiyunsong/UMA-Net/assets/115675554/d062eda9-c2c4-49c8-a262-0b0d5bee296c)




## 1.create conda environment.
`conda create -n UMA python=3.8 -y`

`conda activate UMA`

`conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y`

`pip install  opencv-python pytorch-msssim==0.2.1 timm==0.5.4 tqdm thop fvcore tensorboardx==2.5.1 `


## 2.Clone repository.
`git clone https://github.com/weiyunsong/UMA-Net.git`

## 3 training and test

For example, 

traning

`python train.py --model MA-2 --model_name MA.py --save_dir ./result --datasets_dir ./data --train_dataset ITS --valid_dataset SOTS --exp_config indoor --exp_name training`

test

`python test.py --model MA-2 --model_weight ./result/RESIDE-IN/MA-2/MA-2.pth --data_dir ./data --save_dir ./result --dataset SOTS-- subset indoor`
