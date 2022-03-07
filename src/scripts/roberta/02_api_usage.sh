CUDA_VISIBLE_DEVICES=0 python -u train_roberta.py \
    --name api-usage \
    --train_data ../data/train/usage \
    --test_data ../data/test/usage \
    --save_folder ./results/roberta/api-usage 2>&1| tee ./logs/roberta/02-api-usage.log