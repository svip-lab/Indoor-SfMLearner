CUDA_VISIBLE_DEVICES=1 python evaluation/nyuv2_eval_depth.py \
    --data_path ./data/nyu_test/ \
    --load_weights_folder ./ckpts/weights_5f/ \
    --post_process
