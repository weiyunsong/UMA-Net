A light-weight multiscale adaptive network for single image dehazing 

1.create conda environment.
conda create -n UMA python=3.8 -y
conda activate UMA
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install  opencv-python pytorch-msssim==0.2.1 timm==0.5.4 tqdm thop fvcore tensorboardx==2.5.1 


2.Clone repository.
