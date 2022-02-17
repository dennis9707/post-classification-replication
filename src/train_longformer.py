import sys
sys.path.append("../")
import numpy as np
from datetime import datetime
import gc
import logging
import os
import pandas as pd
import torch
from data_structure.structure import QuestionDataset
from util.util import save_check_point, load_check_point
from util.eval_util import evaluate_batch
from util.data_util import load_data_to_dataset, get_dataloader, get_distribued_dataloader
from model.loss import loss_fn
from train import get_optimizer_scheduler,get_train_args, init_train_env



logger = logging.getLogger(__name__)
def get_exe_name(args):
    exe_name = "{}_{}_{}"
    time = datetime.now().strftime("%m-%d %H-%M-%S")

    base_model = ""
    if args.model_path:
        base_model = os.path.basename(args.model_path)
    return exe_name.format(args.tbert_type, time, base_model)

def log_train_info(args):
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

    args = get_train_args()
    model = init_train_env(args)

    epoch_batch_num = args.train_numbers / args.train_batch_size
    t_total = epoch_batch_num // args.gradient_accumulation_steps * args.num_train_epochs

    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)
    # get the name of the execution
    exp_name = get_exe_name(args)
    # make output directory
    args.output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("n_gpu: {}".format(args.n_gpu))
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))
        logger.info("model loaded")
    log_train_info(args)
    args.global_step = 0
    
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    for epoch in range(args.num_train_epochs):
        logger.info(
                '############# Epoch {}: Training Start   #############'.format(epoch)) 
        # Load dataset and dataloader
        training_set = QuestionDataset(args.binary_type)
        
        if args.local_rank == -1:
            train_data_loader = get_dataloader(
                train_dataset, args.train_batch_size)
        else: 
            train_data_loader = get_distribued_dataloader(
                train_dataset, args.train_batch_size)          
        tr_loss = 0
        model.train()
        model.zero_grad()
        for step, data in enumerate(train_data_loader):
            text_ids = data['text_ids'].to(args.device, dtype=torch.long)
            text_mask = data['text_mask'].to(args.device, dtype=torch.long)
            targets = data['labels'].to(args.device, dtype=torch.float)
            outputs = model(text_ids=text_ids,
                            text_attention_mask=text_mask)

            loss = loss_fn(outputs, targets)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                args.global_step += 1

                if args.logging_steps > 0 and args.global_step % args.logging_steps == 0:
                    tb_data = {
                        'lr': scheduler.get_last_lr()[0],
                        'loss': tr_loss / args.logging_steps
                    }
                    logger.info("tb_data {}".format(tb_data))
                    logger.info(
                        'Epoch: {}, Batch: {}， Loss:  {}'.format(epoch, step, tr_loss / args.logging_steps))
                    tr_loss = 0.0
        
        # validation
        
        
        # Save model checkpoint Regularly
        model_output = os.path.join(
            args.output_dir, "final_model-{}".format(file_cnt))
        save_check_point(model, model_output, args,
                        optimizer, scheduler)


if __name__ == "__main__":
    main()