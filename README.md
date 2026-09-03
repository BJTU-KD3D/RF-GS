# RF-GS

Official implementation of **RF-GS: Reducing Floaters with High-order
Regularization for 3D Gaussian Splatting**.

RF-GS introduces opacity-aware regularization and pruning to suppress abnormal
high-opacity Gaussian points and reduce floater artifacts. The implementation
is provided as a compact patch for Pixel-GS so that the upstream project,
submodules, viewer, and data-processing tools remain unchanged.

## Contents

| File | Destination in Pixel-GS |
| --- | --- |
| `train.py` | `train.py` |
| `rf_pruning.py` | `rf_pruning.py` |
| `gaussian_model.py` | `scene/gaussian_model.py` |
| `loss_utils.py` | `utils/loss_utils.py` |
| `arguments.py` | `arguments/__init__.py` |
| `dataset_readers.py` | `scene/dataset_readers.py` |
| `run.sh` | `run.sh` |

The release contains source code only. Datasets, checkpoints, logs, rendered
images, point clouds, and experiment outputs are intentionally excluded.

## Installation

Clone Pixel-GS with its submodules and create the environment described by the
upstream project:

```bash
git clone --recursive https://github.com/zhengzhang01/Pixel-GS.git
cd Pixel-GS

conda create -n rfgs python=3.9 -y
conda activate rfgs
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
cd ..
```

Apply the RF-GS patch. The installer validates the Pixel-GS checkout and saves
the replaced files under `Pixel-GS/.rf-gs-backup/`.

```bash
pip install -r RF-GS/requirements.txt
bash RF-GS/move_files_to_pixelgs.sh ./Pixel-GS
cd Pixel-GS
```

## Datasets

RF-GS uses the same COLMAP-based scene format as Pixel-GS and 3DGS.

- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/): combine the scenes from
  Dataset Parts 1 and 2 after downloading them.
- [Tanks and Temples](https://www.tanksandtemples.org/download/): follow the
  Pixel-GS preprocessing instructions or use its processed release.
- [RF-GS Real-World dataset](https://drive.google.com/file/d/1aQb3zRGBIOcqzKTWiBbAKqcf6T1VEI-i/view?usp=sharing).
- Custom scenes: follow the 3DGS
  [COLMAP conversion instructions](https://github.com/graphdeco-inria/gaussian-splatting#processing-your-own-scenes).

A processed scene should have this basic structure:

```text
scene/
├── images/
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Training

### Coarse Pruning

1. Run opacity-based coarse pruning:

   ```shell
   bash run.sh /path/to/scene /path/to/output 1
   ```

2. Run score-based coarse pruning using LightGaussian:

   ```shell
   python prune_finetune.py
   ```

### Fine Pruning

Run KD-tree-based fine pruning:

```shell
python rf_pruning.py
```

## Evaluation

```shell
python render.py -m /path/to/output --skip_train
python metrics.py -m /path/to/output
```

### Sparse-view DTU

`dataset_readers.py` supports fixed training views and an external initial
point cloud through environment variables. For the nine-view CoR-GS protocol:

```bash
PIXELGS_TRAIN_INDICES="25,22,28,40,44,48,0,8,13" \
PIXELGS_INIT_PLY="/path/to/scan/9_views/dense/fused.ply" \
python train.py -s /path/to/scan -m ./output/scan_9views --eval
```

## Viewer

RF-GS keeps the Pixel-GS/3DGS point-cloud format and can use the original 3DGS
viewer. See the upstream
[viewer documentation](https://github.com/graphdeco-inria/gaussian-splatting#interactive-viewers).

## TODO List

- [ ] Update the score-based coarse-pruning implementation and complete the
  geometry-and-anisotropy fine-pruning implementation.
- [ ] Provide a demo and additional visualizations.

## Acknowledgements and License

This code is based on Pixel-GS and 3D Gaussian Splatting. Please cite those
projects when appropriate. The inherited source files retain their original
copyright headers and are subject to `LICENSE_3DGS.md`; repository-level terms
are provided by the root `LICENSE.md`.
