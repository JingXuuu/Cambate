# Cambate: Video <ins>C</ins>aptioning with M<ins>amba</ins>-<ins>T</ins>ransform<ins>e</ins>r

Authors: Ng Jing Xu



## Overview

This is a project of 2024 PKU CVDL course. This project uses [Mamba](https://arxiv.org/abs/2312.00752) as an encoder, [Transformer](https://arxiv.org/abs/1706.03762) as a decoder, [ActivityNet captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/) as dataset to generate caption for videos. 



## Prerequisites

#### Environment

- Can only run on linux with nvidia gpu.

- Create environment with `conda env create -f environment.yml`
- You can test whether the models are working in good condition by running `demo.py`

#### Data preparation (It is needed for inference too)

- Download the C3D video features from [here](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav/folder/137471953557) 
- Move the downloaded raw data into folder `data/activitynet-captions`
- Run `unzip.sh`



## Run model

#### Train

1. set the configurations of the model in config.py
2. python3 train.py

#### Generate caption

1. In `gen.sh`, choose the video from train or test dataset. (Self-upload video is not supported)
2. Run `bash gen.sh`
3. Generated captions and ground truth captions with timestamps are shown in `output.txt`
4. Generated captions are also written in `visualization/results.json` for visualization

#### Visualization

1. Use the generated link to download the video (If the video is lost, change another video index)
2. Move the video to the visualization/ folder, and change the video name into 'video.mp4'
3. Run `bash vis.sh`
4. The video with captions are shown in `output.mp4`
5. Sample results are shown in [here](https://drive.google.com/drive/folders/13sFmIZVGUYXS3KDnyWpgT5COAmrhlL86?usp=sharing)

#### Results
1. 3 sentences are generated for every 512 frames
2. These 3 sentences split the timestamp evenly
	

## Weights download

- Download the weights of encoder and decoder [here](https://drive.google.com/drive/folders/13sFmIZVGUYXS3KDnyWpgT5COAmrhlL86?usp=sharing) (only first, last and best epoch weights are uploaded)
- Place the weights into folder `ckpt/`



## Problems

- Not equipped with conv3d net, unable to upload own video and generation caption onto it.

- The process of generating caption is complicated as incapable to download Youtube videos directly.