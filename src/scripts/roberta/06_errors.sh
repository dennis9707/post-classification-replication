CUDA_VISIBLE_DEVICES=5 python -u train_roberta.py \
    --name errors \
    --train_data ../data/train/errors \
    --test_data ../data/test/errors \
    --save_folder ./results/roberta/errors 2>&1| tee ./logs/roberta/06-errors.log