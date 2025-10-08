# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.utils.rnn import pad_sequence
# from transformers import AutoModel, EnergyUsageTracker
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

####AERO Model
from models.DGCPath import DGCPath as DGCPath
from models.DGCPath import MultiTaskLossWrapper as MTW
from engine_pretrain_dgcpath import train_one_epoch

import pickle as pkl

def get_args_parser():
    parser = argparse.ArgumentParser('PRL pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Temperature (default: 0.05)') 
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-6, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256') 
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--nlayers', type=int, default=12, metavar='N',
                        help='layers')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--checkpoints-path', default='./Results/DGCPath-CD-', type=str) 
    parser.add_argument('--timesave', default='DGCPath-', type=str)
    parser.add_argument('--modename', default='DGCPath-', type=str)

    parser.add_argument('--distributed', action='store_false', dest='no-distributed')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',help='url used to set up distributed training')

    return parser

node2vec = pkl.load(open('../Data/road_network_chengdu_220115_128.pkl','rb')) ###512
node2vec = torch.FloatTensor(node2vec)


def collate_fn(batch):
    paths, labels = zip(*batch)
    padded_paths = pad_sequence(paths, batch_first=True, padding_value=3814)  # 默认 padding 0 8497 for Harbin
    labels = torch.stack(labels)
    return padded_paths, labels

class TrainDataset(Dataset):
    def __init__(self):
        ####Path, Time, Dis, OD--->TT  Path, Sim, OD---->PR
        path,yc,_,_= pkl.load(open('../Data/CD/AXXX.pkl', 'rb'))
        self.path = [torch.tensor(p, dtype=torch.long) if isinstance(p, (list, tuple)) else torch.tensor([p], dtype=torch.long) for p in path]
        self.yc = torch.tensor(np.asarray(yc, dtype=np.int64), dtype=torch.long)
        self.len = len(self.path)
    def __getitem__(self,index):
        return self.path[index], self.yc[index]
    def __len__(self):
        return self.len

def translate_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'module' in key:
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def main(args):
    # misc.init_distributed_mode(args)

    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    dataset_train = TrainDataset()

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,collate_fn=collate_fn
    )

    multi_task_loss = MTW()
    model = DGCPath( 
            node2vec = node2vec,
            input_dim=128,
            hidden_dim=512,
            latent_dim=128,
            output_dim=128,
            in_chans=1,
            embed_dim = 128,
            depth = args.nlayers,
            num_heads = 8,
            Temp=args.temp,
        ) 
    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups.append({'params': multi_task_loss.parameters(), 'weight_decay': 0.05})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            state_dict = translate_state_dict(model.state_dict())
            total_time1=0
            state_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                }
            torch.save(
                        state_dict,
                        args.checkpoints_path + '/' + args.modename+'_Layer'+str(args.nlayers)+'_'+str(epoch) + '.pth') ###checkpoints
  

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "DGCPath-"+str(args.nlayers)+"-CD.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
