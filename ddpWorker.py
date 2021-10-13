import torch
import argparse
import numpy as np
import random
from scipy import stats
import math

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import module as md
import data_loader
import spix_rim as sp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, default='koniq-10k',
                    help='Database: livec|koniq-10k|live-fb|live|csiq')
parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                    help='Number of sample patches from training image')
parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                    help='Number of sample patches from testing image')
parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=100,
                    help='lr ratio for precisely controlling holistic lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=24, help='Batch size in every gpu')
parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs for training and testing')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=432,
                    help='Crop size for training or testing image patches')
parser.add_argument('--seed', dest='seed', type=int, default=1, help='Control same seed')
parser.add_argument('--local_rank', dest='local_rank', type=int, default=-1)


def main():
    config = parser.parse_args()
    folder_path = {
        'live': '../databaserelease2/',
        'csiq': '../CSIQ/',
        'livec': '../ChallengeDB_release/',
        'koniq-10k': '../koniq-10k/',
        'live-fb': '../database/'
    }
    img_num = {
        'live': list(range(0, 29)),
        'csiq': list(range(0, 30)),
        'livec': list(range(0, 1162)),
        'koniq-10k': list(range(0, 10073)),
        'live-fb': list(range(0, 37980))
    }
    gpu = config.local_rank
    lr = config.lr
    lrratio = config.lr_ratio

    init_seed(config.seed, gpu)

    torch.cuda.set_device(gpu)  # set current gpu
    dist.init_process_group(backend='nccl', world_size=4)
    device = torch.device("cuda", gpu)
    model_ms = md.DSNet(16, 224).to(device)  # Apply models to gpu or cpu
    model_sps = sp.CNNRIM(5, 100, 32, 5).to(device)
    model_fc = md.PredictNet().to(device)

    model_ms = DDP(model_ms, device_ids=[gpu], output_device=gpu)  # Apply DDP, and copy parameters from device[0]
    model_fc = DDP(model_fc, device_ids=[gpu], output_device=gpu)

    backbone_params = list(map(id, model_ms.module.res.parameters()))  # Extract Resnet parameters
    msnet_params = list(filter(lambda p: id(p) not in backbone_params, model_ms.parameters()))  # Extract the Rest parameters
    paras = [{'params': msnet_params, 'lr': lr * lrratio},
             {'params': model_ms.module.res.parameters(), 'lr': lr},
             {'params': model_fc.parameters(), 'lr': lr * lrratio}
             ]
    l1_loss = torch.nn.L1Loss().to(device)
    solver = torch.optim.Adam(paras, weight_decay=config.weight_decay)

    # Load DATA
    sel_num = img_num[config.dataset]
    random.shuffle(sel_num)  # make index random
    train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
    test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]

    path = folder_path[config.dataset]
    train_dataset = data_loader.DataLoader(config.dataset, path, train_index, config.patch_size, config.train_patch_num,
                                           batch_size=config.batch_size, istrain=True, isFull=True)
    test_dataset = data_loader.DataLoader(config.dataset, path, test_index, config.patch_size, config.test_patch_num,
                                          istrain=False, isFull=True)
    # allocate data for each gpu
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset.data, drop_last=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset.data, shuffle=False, drop_last=True)

    train_data = train_dataset.get_data(train_sampler)  # use DataLoader
    test_data = test_dataset.get_data(test_sampler)
    if dist.get_rank() == 0:
        print('Training and testing on %s dataset for %d epochs' % (config.dataset, config.epochs))
        print('Epoch\tTrain_Loss\t\tTrain_SRCC\tTest_SRCC\t\tTest_PLCC', flush=True)

    # Start to train and test
    best_srcc = 0.0
    best_plcc = 0.0
    model_ms.train(True)
    model_fc.train(True)
    model_sps.train(True)
    flag = False
    for t in range(config.epochs):
        epoch_loss = []
        pred_scores = []
        gt_scores = []
        train_sampler.set_epoch(t)
        for img, label in train_data:
            img = img.to(device)
            label = label.to(device)
            num = list(img.size())
            for n in range(num[0]):  # generate superpixel probability map
                oneimg = img[n]
                spix = model_sps.optimize(oneimg, 5, 1e-2, 2, 2, 2, "cuda")
                if n == 0:
                    all_spix = spix
                else:
                    all_spix = torch.cat((all_spix, spix), 0)
            solver.zero_grad()
            paras = model_ms(img, all_spix)
            pred = model_fc(paras['fe_in_vec'])
            pred_scores.append(pred.data)
            gt_scores.append(label.data)

            loss = l1_loss(pred.squeeze(), label.float().detach())
            epoch_loss.append(loss.item())
            loss.backward()
            solver.step()
        torch.distributed.barrier()
        pred_scores = distributed_concat(torch.cat(pred_scores, dim=0))  # gather all results from each process
        gt_scores = distributed_concat(torch.cat(gt_scores, dim=0))
        epoch_loss = distributed_concat(torch.tensor(epoch_loss).to(device))
        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_srcc, test_plcc = test(test_data, model_ms, model_fc, model_sps, device, config)

        if dist.get_rank() == 0 and test_srcc > best_srcc:
            best_srcc = test_srcc
            best_plcc = test_plcc
        if dist.get_rank() == 0:
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc), flush=True)
        if t == 15:
            lrratio = 10
            flag = True
        if t == 60:
            lr = 2e-6
            flag = True
        paras = [{'params': msnet_params, 'lr': lr * lrratio},
                 {'params': model_ms.module.res.parameters(), 'lr': lr},
                 {'params': model_fc.parameters(), 'lr': lr * lrratio}
                 ]
        if flag:
            solver = torch.optim.Adam(paras, weight_decay=config.weight_decay)
            flag = False
    if dist.get_rank() == 0:
        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc), flush=True)


def distributed_concat(tensor, reshape=False):  # gather result and return a list
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    if reshape:
        for i in range(len(output_tensors)):
            output_tensors[i] = output_tensors[i].view(-1, 6)
        concat = torch.cat((output_tensors), dim=1).cpu().tolist()
    else:
        concat = torch.cat(output_tensors, dim=0).cpu().tolist()
    return concat


def test(data, model_ms, model_fc, model_sps, device, config):
    model_ms.train(False)
    model_fc.train(False)
    pred_scores = []
    gt_scores = []
    if config.test_patch_num == 1:
        reshape = False
    else:
        reshape = True
    for img, label in data:
        img = img.to(device)
        label = label.to(device)
        spix = model_sps.optimize(img.squeeze(0), 5, 1e-2, 2, 2, 2, "cuda")
        paras = model_ms(img, spix)
        pred = model_fc(paras['fe_in_vec'])
        pred_scores.append(float(pred.item()))
        gt_scores.append(label.item())
    torch.distributed.barrier()
    pred_scores = distributed_concat(torch.tensor(pred_scores).to(device), reshape=reshape)
    gt_scores = distributed_concat(torch.tensor(gt_scores).to(device), reshape=reshape)
    if reshape:
        pred_scores = np.mean(np.array(pred_scores), axis=1)
        gt_scores = np.mean(np.array(gt_scores), axis=1)

    test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
    test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

    model_ms.train(True)
    model_fc.train(True)
    return test_srcc, test_plcc


def init_seed(seed, local_rank):
    random.seed(seed)
    np.random.seed(seed + local_rank)
    # torch.manual_seed(seed)
    if local_rank == 0:
        print('The seed is %d' % seed)


if __name__ == '__main__':
    main()
