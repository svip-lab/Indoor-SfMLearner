CUDA_VISIBLE_DEVICES=1 python evaluation/scannet_eval_pose.py \
    --data_path ./data/scannet_pose \
    --load_weights_folder ./ckpts/weights_5f/ \
    --frame_ids 0 1
