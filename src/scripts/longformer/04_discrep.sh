CUDA_VISIBLE_DEVICES=3 python -u train_lf_classification.py \
    --name discrep \
    --train_data ../data/train/discrep \
    --test_data ../data/test/discrep \
    --save_folder ./results/discrep 2>&1| tee ./04-disc.log