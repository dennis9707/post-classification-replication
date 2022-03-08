CUDA_VISIBLE_DEVICES=5 python -u train_lf_classification.py \
    --name errors \
    --train_data ../data/train/errors \
    --test_data ../data/test/errors \
    --save_folder ./results/errors 2>&1| tee ./06-errors.log