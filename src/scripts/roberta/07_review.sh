CUDA_VISIBLE_DEVICES=6 python -u train_roberta.py \
    --name review \
    --train_data ../data/train/review \
    --test_data ../data/test/review \
    --save_folder ./results/roberta/review 2>&1| tee ./logs/roberta/07-review.log