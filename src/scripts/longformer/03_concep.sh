CUDA_VISIBLE_DEVICES=2 python -u train_lf_classification.py \
    --name concep \
    --train_data ../data/train/concep \
    --test_data ../data/test/concep \
    --save_folder ./results/concep 2>&1| tee ./03-concep.log