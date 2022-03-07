CUDA_VISIBLE_DEVICES=5 python -u train_codebert.py \
    --name errors \
    --train_data ../data/train/errors \
    --test_data ../data/test/errors \
    --save_folder ./results/errors 2>&1| tee ./logs/codebert/06-errors.log