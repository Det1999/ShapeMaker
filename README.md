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

When using the full target point cloud as input, the training is divided into four stages, namely training Module Canonicalization (1), training Module Segmentation (2), training Module Deformation (3), and training Module Retrieval (4). When using Class chair, the specifics are as follows:
```
1. python tools/full/train_full.py -c configs/Tables.yaml
2. python tools/full/train_rd_kp_seg.py -c configs/Tables.yaml
3. python tools/full/cage_deform.py -c configs/Tables.yaml
4. python tools/full/cage_retrieval.py -c configs/Tables.yaml
```

To train the module on the chair category with input of partial target point clouds run:
```
1. python tools/partial/train_full.py
2. python tools/partial/train_rd_kp_seg.py
3. python tools/partial/train_partial_rd.py
4. python tools/partial/cage_deform_partial.py
5. python tools/partial/cage_retrieval_partial.py
```


