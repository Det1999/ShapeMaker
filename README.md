# ShapeMatcher: Self-Supervised Joint Shape Canonicalization, Segmentation, Retrieval and Deformation, CVPR2024
Yan Di, Chenyangguang Zhang, Chaowei Wang, Ruida Zhang, Guangyao Zhai, Yanyan Li, Bowen Fu, Xiangyang Ji and Shan Gao

## Install

Install using [conda](https://docs.conda.io/en/latest/):
```
conda env create -f environment.yml 
conda activate ShapeMaker
```

## Training
Download [ShapeNet](https://shapenet.org/download/shapenetcore) to `datasets/ShapeNetCore.v1`.

When using the full target point cloud as input, the training is divided into four stages, namely training Module Canonicalization (1), training Module Segmentation (2), training Module Deformation (3), and training Module Retrieval (4). When using category tables, the specifics are as follows:
```
1. python tools/full/train_full.py -c configs/Tables.yaml
2. python tools/full/train_rd_kp_seg.py -c configs/Tables.yaml
3. python tools/full/cage_deform.py -c configs/Tables.yaml
4. python tools/full/cage_retrieval.py -c configs/Tables.yaml
```

When using partial point clouds as input, the training is divided into five stages, which adds consistency learning between the partial branch and the full branch (3) compared to when using the full point cloud as input. When using category tables, the specifics are as follows:
```
1. python tools/partial/train_full.py -c configs/Tables.yaml
2. python tools/partial/train_rd_kp_seg.py -c configs/Tables.yaml
3. python tools/partial/train_partial_rd.py -c configs/Tables.yaml
4. python tools/partial/cage_deform_partial.py -c configs/Tables.yaml
5. python tools/partial/cage_retrieval_partial.py -c configs/Tables.yaml
```
If you want to train using other categories, besides changing the config file, you may need to modify the names of the weights in the training file.

## Testing
To test the trained R&D model on full or partial point clouds run:
```
python tools/cage_EVAL.py -c configs/Tables.yaml -t configs/test.yaml 
```
During testing, you may need to modify the hyperparameter RATE in the test file, as well as the corresponding weight names to be loaded.

## Checkpoint
We have released the checkpoint we obtained during our training process. You can click the link to download: [ShapeMatcher](https://drive.google.com/drive/folders/1JcYgRoZq2QyTYLw0JQY_UseelfzenukJ?usp=drive_link).

