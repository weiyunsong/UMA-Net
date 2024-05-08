A light-weight multiscale adaptive network for single image dehazing 
![UMA-net](https://github.com/weiyunsong/UMA-Net/assets/115675554/f179fc13-b0ee-4d7d-8aa7-fb966c91c14d)

1.create conda environment.
conda create -n UMA python=3.8 -y
conda activate UMA
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install  opencv-python pytorch-msssim==0.2.1 timm==0.5.4 tqdm thop fvcore tensorboardx==2.5.1 


2.Clone repository.
git clone https://github.com/weiyunsong/UMA-Net.git

3 training and test

For example, 

traning

python train.py --model MA-2 --model_name MA.py --save_dir ./result --datasets_dir ./data --train_dataset ITS --valid_dataset SOTS --exp_config indoor --exp_name training

test

python test.py --model MA-2 --model_weight ./result/RESIDE-IN/MA-2/MA-2.pth --data_dir ./data --save_dir ./result --dataset SOTS-- subset indoor
