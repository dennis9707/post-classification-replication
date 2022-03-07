CUDA_VISIBLE_DEVICES=4 python -u train_roberta.py \
    --name docs \
    --train_data ../data/train/docs \
    --test_data ../data/test/docs \
    --save_folder ./results/roberta/docs 2>&1| tee ./logs/roberta/05-docs.log