CUDA_VISIBLE_DEVICES=1 python evaluation/scannet_eval_depth.py \
    --data_path ./data/scannet_test \
    --load_weights_folder ./ckpts/weights_5f/ \
    --post_process
