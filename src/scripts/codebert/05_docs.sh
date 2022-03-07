CUDA_VISIBLE_DEVICES=4 python -u train_codebert.py \
    --name docs \
    --train_data ../data/train/docs \
    --test_data ../data/test/docs \
    --save_folder ./results/docs 2>&1| tee ./logs/codebert/05-docs.log