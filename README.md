# RF-GS: Reducing Floaters with High-order Regularization for 3D Gaussian Splatting

This is the official code repository for the paper:

**"RF-GS: Reducing Floaters with High-order Regularization for 3D Gaussian Splatting"**

---

## Dataset

The Real-World(RW) dataset can be downloaded from Google Drive:

[Download RW Dataset](https://drive.google.com/file/d/1aQb3zRGBIOcqzKTWiBbAKqcf6T1VEI-i/view?usp=sharing)

After downloading and extracting, place the dataset as:

```text
dataset/
├── Bike/
│   ├── images/
│   └── sparse/0/
├── Building/
│   ├── images/
│   └── sparse/0/
├── Playground/
│   ├── images/
│   └── sparse/0/
├── Podium/
│   ├── images/
│   └── sparse/0/
├── indoor/
│   ├── Bar/
│   ├── Chair/
│   ├── Coffee/
│   ├── Dormitory/
│   └── Sofa/
    |---...
```

### Mip-NeRF 360 Dataset

Please download the Mip-NeRF 360 dataset processed by colmap from [Mip-NeRF 360](https://jonbarron.info/mipnerf360/):

```
360_v2
    |---bicycle
    |   |---images
    |   |   |---<image 0>
    |   |   |---<image 1>
    |   |   |---...
    |   |---images_2
    |   |---images_4
    |   |---images_8
    |   |---sparse
    |       |---0
    |           |---cameras.bin
    |           |---images.bin
    |           |---points3D.bin
    |---bonsai
    |---...
```


### Tanks and Temples Dataset

#### Option 1

We thank [Pixel-GS](https://github.com/zhengzhang01/Pixel-GS) for constructing the processed Tanks and Temples dataset, which is available for direct download via [OneDrive](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/u3009782_connect_hku_hk/EehzMcKeoclAnVdgPyyBxNwB24ve5bk3ZSct38AUWPbprw?e=uWEc5a). Please agree the official license before download it.

#### Option 2 

Tanks and Temples is divided into three parts, comprising a total of 21 scenes: Intermediate ('Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train'), Advanced ('Auditorium', 'Ballroom', 'Courtroom', 'Museum', 'Palace', 'Temple'), and Training Data ('Barn', 'Caterpillar', 'Church', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck').

Please download the "image set" of all scenes from the Tanks and Temples dataset from [Tanks and Temples](https://www.tanksandtemples.org/download/). After unzipping, rename the image folder directories of all scenes to "input". The organized folder structure is as follows:

```
---tanks_and_temples
    |---Auditorium
    |   |---input
    |   |   |---<image 0>
    |   |   |---<image 1>
    |   |   |---...
    |---Ballroom
    |---...
```

After configuring libraries such as colmap according to the method in the original [Pixel-GS](https://github.com/zhengzhang01/Pixel-GS), use the following command to generate camera poses for all scenes in Tanks and Temples:

```
python ./prepose.py
```

Finally, the current directory should contain the following folders:

```
---tanks_and_temples
    |---Auditorium
    |   |---images
    |   |   |---<image 0>
    |   |   |---<image 1>
    |   |   |---...
    |   |---images_2
    |   |---images_4
    |   |---images_8
    |   |---sparse
    |       |---0
    |           |---cameras.bin
    |           |---images.bin
    |           |---points3D.bin
    |---Ballroom
    |---...
```



### Your Own Dataset

Our method requires the same data format as 3DGS. For your own data, you can use the processing method found in the ["Processing your own Scenes"](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) section of the original 3DGS code.

## Getting Started 

Our code is based on the excellent official repo for [Pixel-GS](https://github.com/zhengzhang01/Pixel-GS) and [LightGaussian](https://github.com/VITA-Group/LightGaussian). 

## Training

### Coarse Pruning

1. Run opacity-based coarse pruning:

   ```shell
   bash run.sh 
   ```

2. Run score-based coarse pruning using [LightGaussian](https://github.com/VITA-Group/LightGaussian):

   ```shell
   python prune_finetune.py
   ```

### Fine Pruning

Run KD-tree-based fine pruning:

```shell
python rf_pruning.py
```

## Pre-trained Models






## Acknowledgement
This project is built upon [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), [Pixel-GS](https://github.com/zhengzhang01/Pixel-GS) and [LightGaussian](https://github.com/VITA-Group/LightGaussian). We thank all authors for their great work!
## License

This repository is released under the Apache 2.0 license. Please see the [LICENSE](./LICENSE) file for more information.


