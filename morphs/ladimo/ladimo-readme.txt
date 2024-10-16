LADIMO: Face Morph Generation through Biometric Template Inversion with Latent Diffusion
Preseneted at International Joint Conference on Biometrics 2024

Example Inference Code:

1) Installation

Make sure to install the two conda environments (Python 3.8) stored under ./scripts ("ladimo_env.yml" and "magface_env.yml")  - ie: 

conda env create -f ladimo_env.yml

conda env create -f magface_env.yml


2) Extract MagFace embeddings (done for example images)

Activate magface conda environment:

conda activate magface

Navigate to ./scripts and run:

python get_magface_embeds.py


3) LADIMO face morph Generation

Activate LADIMO conda environment:

conda activate ldm 

Run inference code that morphs the two example images:

python ladimo_inference.py




