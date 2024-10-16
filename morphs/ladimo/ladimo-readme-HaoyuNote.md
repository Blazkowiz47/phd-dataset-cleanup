# LADIMO

This is a simple doc from Haoyu to run LADIMO morphing algorithm， tested on: Ubuntu 20.04.2 LTS， python 3.8.10, A100 with CUDA 11.0

# Requirements
```
sudo pip install -U virtualenv
```
## Magface

```
virtualenv -p python3 magface

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install tqdm==4.66.1
pip install opencv-python
pip install mxnet==1.5.1
pip install termcolor==2.4.0
pip install scikit-image==0.21.0
pip install easydict==1.13
```

## LADIMO


install pytorch


```
virtualenv -p python3 ladimo

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt

```

# How to run

1. extract magface embedding

    1.1 prepare contibuting images, and place it to [reference](./taming-transformers/taming/data/morph_db/reference) folder

    1.2 
    ```
    source PATH_TO_magface_VENV/bin/activate

    cd scripts

    # edit to your need
    python get_magface_embeds_frgc.py

    ```
    embeddings will be extracted at [reference_magface](./taming-transformers/taming/data/morph_db/reference_magface) folder

2. run LADIMO to generate morphs

    2.1 prepare morph list as [Morph_LADIMO_P1_Female.txt](./Morph_LADIMO_P1_Female.txt)
    
    2.2 edit [ladimo_inference_frgc.py](./ladimo_inference_frgc.py) to your needs
    ```
    python ladimo_inference_frgc.py

    ```
