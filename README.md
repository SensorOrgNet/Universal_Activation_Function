# Universal Activation Function
Tensorflow and Pytorch source code for the paper 

[Yuen, Brosnan, Minh Tu Hoang, Xiaodai Dong, and Tao Lu. "Universal activation function for machine learning." Scientific reports 11, no. 1 (2021): 1-11.](https://www.nature.com/articles/s41598-021-96723-8)


# Getting the code

Use git to pull this repo
```
git clone https://github.com/SensorOrgNet/Universal_Activation_Function.git
```


# Running the Pytorch version


Requires [Docker](https://docs.docker.com/get-docker/) 


Install pytorch container
```
docker run --name UAF --gpus all  -v /home/username/UAF/:/workspace  -w /workspace    -it  nvcr.io/nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04   bash
```

Install python
```
apt update
apt install python3-pip
```

Install pytorch and pytorch geometric
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip3 install torch-cluster -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip3 install torch-spline-conv -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip3 install torch-geometric
```


Run the GCN2 with UAF for CORA dataset. The fold number is represented by the number at the end
```
cd   Universal_Activation_Function/pytorch/
python3   ./gcn2_cora_UAF.py  0
```





Run the PNA with UAF for ZNC dataset. The fold number is represented by the number at the end
```
cd   Universal_Activation_Function/pytorch/
python3   ./pna_UAF.py  0
```



