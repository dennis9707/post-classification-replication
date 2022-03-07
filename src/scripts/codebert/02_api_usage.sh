CUDA_VISIBLE_DEVICES=0 python -u train_codebert.py \
    --name api-usage \
    --train_data ../data/train/usage \
    --test_data ../data/test/usage \
    --save_folder ./results/api-usage 2>&1| tee ./logs/codebert/02-api-usage.log