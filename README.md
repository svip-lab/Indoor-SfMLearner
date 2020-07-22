# Indoor SfMLearner

PyTorch implementation of our ECCV2020 paper:

[P<sup>2</sup>Net: Patch-match and Plane-regularization for Unsupervised Indoor Depth Estimation](https://arxiv.org/pdf/2007.07696.pdf)

Zehao Yu\*,
Lei Jin\*,
[Shenghua Gao](http://sist.shanghaitech.edu.cn/sist_en/2018/0820/c3846a31775/page.htm)

(\* Equal Contribution)

<img src="asserts/pipeline.png" width="800">

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```
Then install pytorch with
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
Pytorch version >= 0.4.1 would work well.

### Download pretrained model
Please download pretrained model from [Onedrive](https://onedrive.live.com/?authkey=%21ANXK7icE%2D33VPg0&id=C43E510B25EDDE99%21106&cid=C43E510B25EDDE99) and extract:
```bash
tar -xzvf ckpts.tar.gz 
rm ckpts.tar.gz
```

### Prediction on single image                                                                                  
Run the following command to predict on a single image:
```bash
python inference_single_image.py --image_path=/path/to/image
```
By default, the script saves the predicted depth to the same folder

## Evaluation                                                                                                     
Download testing data from [Onedrive](https://onedrive.live.com/?authkey=%21ANXK7icE%2D33VPg0&id=C43E510B25EDDE99%21106&cid=C43E510B25EDDE99) and put to ./data.
```bash
cd data
tar -xzvf nyu_test.tar.gz 
tar -xzvf scannet_test.tar.gz
tar -xzvf scannet_pose.tar.gz
cd ../
```

### NYUv2 Dpeth
```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/nyuv2_eval_depth.py \
    --data_path ./data \
    --load_weights_folder ckpts/weights_5f \
    --post_process  
```

### NYUv2 normal
```base
CUDA_VISIBLE_DEVICES=1 python evaluation/nyuv2_eval_norm.py \
    --data_path ./data \
    --load_weights_folder ckpts/weights_5f \
    # --post_process
```

### ScanNet Depth
```base
CUDA_VISIBLE_DEVICES=1 python evaluation/scannet_eval_depth.py \                                               
    --data_path ./data/scannet_test \
    --load_weights_folder ckpts/weights_5f \
    --post_process
```

### ScanNet Pose
```base
CUDA_VISIBLE_DEVICES=1 python evaluation/scannet_eval_pose.py \
    --data_path ./data/scannet_pose \
    --load_weights_folder ckpts/weights_5f \
    --frame_ids 0 1
```

## Training
First download [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) on the official website and unzip the raw data to DATA_PATH.

### Extract Superpixel
Run the following command to extract superpixel:
```bash
python extract_superpixel.py --data_path DATA_PATH --output_dir ./data/segments
```

### 3-frames
Run the following command to train our network:
```bash
CUDA_VISIBLE_DEVICES=1 python train_geo.py \                                                                   
    --model_name 3frames \
    --data_path DATA_PATH \
    --val_path ./data \
    --segment_path ./data/segments \
    --log_dir ./logs \
    --lambda_planar_reg 0.05 \
    --batch_size 12 \
    --scales 0 \
    --frame_ids_to_train 0 -1 1
```

### 5-frames
Using the pretrained model from 3-frames setting gives better results.
```bash
CUDA_VISIBLE_DEVICES=1 python train_geo.py \                                                                   
    --model_name 5frames \
    --data_path DATA_PATH \
    --val_path ./data \
    --segment_path ./data/segments \
    --log_dir ./logs \
    --lambda_planar_reg 0.05 \
    --batch_size 12 \
    --scales 0 \
    --load_weights_folder FOLDER_OF_3FRAMES_MODEL \
    --frame_ids_to_train 0 -2 -1 1 2
```

## Acknowledgements
This project is built upon [Monodepth2](https://github.com/nianticlabs/monodepth2). We thank authors of Monodepth2 for their great work and repo.

## License
TBD

## Citation
Please cite our paper for any purpose of usage.
```
@inproceedings{IndoorSfMLearner,
  author    = {Zehao Yu and Lei Jin and Shenghua Gao},
  title     = {P$^{2}$Net: Patch-match and Plane-regularization for Unsupervised Indoor Depth Estimation},
  booktitle = {ECCV},
  year      = {2020}
}
```

