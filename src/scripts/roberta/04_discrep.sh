CUDA_VISIBLE_DEVICES=3 python -u train_roberta.py \
    --name discrep \
    --train_data ../data/train/discrep \
    --test_data ../data/test/discrep \
    --save_folder ./results/roberta/discrep 2>&1| tee ./logs/roberta/04-disc.log