# ShapeMaker

## Install

Install using [conda](https://docs.conda.io/en/latest/):
```
conda env create -f environment.yml 
conda activate ShapeMaker
```

## Training
Download [ShapeNet](https://shapenet.org/download/shapenetcore) to `datasets/ShapeNetCore.v1`.

To train the module on the table category with input of full target point clouds run:
```
1. python tools/full/train_full.py
2. python tools/full/train_rd_kp_seg.py
3. python tools/full/cage_deform.py
4. python tools/full/cage_retrieval.py
```

To train the module on the chair category with input of partial target point clouds run:
```
1. python tools/partial/train_full.py
2. python tools/partial/train_rd_kp_seg.py
3. python tools/partial/train_partial_rd.py
4. python tools/partial/cage_deform_partial.py
5. python tools/partial/cage_retrieval_partial.py
```


