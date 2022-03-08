CUDA_VISIBLE_DEVICES=6 python -u train_lf_classification.py \
    --name review \
    --train_data ../data/train/review \
    --test_data ../data/test/review \
    --save_folder ./results/review 2>&1| tee ./07-review.log