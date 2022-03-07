CUDA_VISIBLE_DEVICES=6 python -u train_codebert.py \
    --name review \
    --train_data ../data/train/review \
    --test_data ../data/test/review \
    --save_folder ./results/review 2>&1| tee ./logs/codebert/07-review.log