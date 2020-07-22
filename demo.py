import os
import sys
import torch
import torch.nn as nn
import argparse
import logging
import yaml
from munch import Munch
import timm
from timm import Dataset, resolve_data_config
from models import apply_test_time_pool
from timm.data import create_loader
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from datetime import datetime
import numpy as np
from utils import *
import time
from collections import OrderedDict


torch.backends.cudnn.benchmark = True


with open('config/save.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
args = Munch(config)
args.prefetcher = not args.no_prefetcher
args.distributed = False
args.device = 'cuda'
args.world_size = 3
args.rank = 0
logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

args.resume = parser.pt_weights

model_raw = timm.create_model('tf_efficientnet_b1', pretrained=True)
model_raw = model_raw.cuda()
model_raw = torch.nn.DataParallel(model_raw)
if args.resume:
    print('raw load')
    model_raw.load_state_dict(torch.load(args.resume).state_dict())

torch.manual_seed(args.seed + args.rank)

val_dir = os.path.join(parser.dataset, '/val')
data_config = resolve_data_config(vars(args), model=model_raw, verbose=args.local_rank == 0)
num_aug_splits = 0
val_dataset = Dataset(val_dir, load_bytes=False, class_map='')


param_count = sum([m.numel() for m in model_raw.parameters()])
logging.info('Model created, param count: %d' % (param_count))

model_raw, test_time_pool = apply_test_time_pool(model_raw, data_config, args)

crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
val_loader = create_loader(
    val_dataset,
    input_size=data_config['input_size'],
    batch_size=args.batch_size,
    is_training=False,
    use_prefetcher=args.prefetcher,
    interpolation=data_config['interpolation'],
    mean=data_config['mean'],
    std=data_config['std'],
    num_workers=args.workers,
    crop_pct=crop_pct,
    pin_memory=args.pin_mem,
    tf_preprocessing=args.tf_preprocessing)


if args.jsd:
    assert num_aug_splits > 1  # JSD only valid with aug splits set
    train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    validate_loss_fn = nn.CrossEntropyLoss()
# elif args.mixup > 0.:
#     # smoothing is handled with mixup label transform
#     train_loss_fn = SoftTargetCrossEntropy()
    validate_loss_fn = nn.CrossEntropyLoss()
elif args.smoothing:
    train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    validate_loss_fn = nn.CrossEntropyLoss()
else:
    train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = train_loss_fn
    

optimizer = create_optimizer(args, model_raw)

eval_metric = args.eval_metric
best_metric = None
best_epoch = None
saver = None
output_dir = ''


def val_epoch(model_raw, val_loader, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model_raw.eval()
    model.eval()
#     model_att.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        end = time.time()
        for i, (inputs, target) in enumerate(val_loader):
            if args.no_prefetcher:
                target = target.cuda()
                inputs = inputs.cuda()
            
            # synthesizing input + generator
            # compute output
            output, _ = model_raw(inputs)
#             output = model_att(inputs_out, output)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0:
                logging.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        i, len(val_loader), batch_time=batch_time,
                        rate_avg=inputs.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    results = OrderedDict(
        top1=round(top1.avg, 4), top1_err=round(100 - top1.avg, 4),
        top5=round(top5.avg, 4), top5_err=round(100 - top5.avg, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    logging.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))
    return results

for epoch in range(args.start_epoch, args.epochs):

    eval_metrics = val_epoch(model_raw, val_loader, validate_loss_fn, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', required=True, help='dataset location')
    parser.add_argument('--pt_weight', required=True, help='pre-trained weights location')