CUDA_VISIBLE_DEVICES=1 python train_geo.py \
    --model_name reg0.01_smooth_0.001_s1 \
    --data_path ./data/ \
    --val_path ./data/nyu_test \
    --segment_path ./data/segments \
    --log_dir ./logs \
    --lambda_planar_reg 0.05 \
    --batch_size 12 \
    --scales 0 \
    --frame_ids_to_train 0 -1 1


