CUDA_VISIBLE_DEVICES=4 python -u train_lf_classification.py \
    --name docs \
    --train_data ../data/train/docs \
    --test_data ../data/test/docs \
    --save_folder ./results/docs 2>&1| tee ./05-docs.log