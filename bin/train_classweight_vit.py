import sys
import os
import argparse
import logging
import json
import time
import subprocess
from shutil import copyfile

import numpy as np
from sklearn import metrics
from easydict import EasyDict as edict
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
import datetime
from tqdm import tqdm



from tensorboardX import SummaryWriter
from torchsummary import summary as SUMMARY
#from torch.utils.tensorboard import SummaryWriter
#from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Primary device

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

#from data.datasetCXP import ImageDataset_CXP
#from data.datasetNIH import ImageDataset_NIH
from data.dataset_Mayo import ImageDataset_Mayo
#from model.classifier import Classifier  # noqa
#from model.backbone.convnext import (convnext_tiny, convnext_base, convnext_small, convnext_large, convnext_xlarge)
from model.classifier_vit import VIT
from utils.misc import lr_schedule  # noqa
from model.utils import get_optimizer  # noqa

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('cfg_path', default=None, metavar='CFG_PATH', type=str,
                    help="Path to the config file in yaml format")
parser.add_argument('save_path', default=None, metavar='SAVE_PATH', type=str,
                    help="Path to the saved models")
parser.add_argument('--num_workers', default=8, type=int, help="Number of "
                    "workers for each data loader")
parser.add_argument('--device_ids', default='0,1,2,3', type=str,
                    help="GPU indices ""comma separated, e.g. '0,1' ")
parser.add_argument('--pre_train', default=None, type=str, help="If get"
                    "parameters from pretrained model")
parser.add_argument('--resume', default=0, type=int, help="If resume from "
                    "previous run")
parser.add_argument('--logtofile', default=False, type=bool, help="Save log "
                    "in save_path/log.txt if set True")
parser.add_argument('--verbose', default=False, type=bool, help="Detail info")


def get_loss(output, target, cfg):
    if cfg.criterion == 'BCE':
        # Ensure shapes match
        output = output.view(-1).to(device)  # [batch_size]
        target = target.view(-1).to(device)  # [batch_size]
        # print(f"target:{target.shape}")
        # print(f"output:{output.shape}")

        # Handle positive class weight
        pos_weight = torch.tensor(cfg.pos_weight, dtype=torch.float32, device=device)

        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                pos_count = target.sum()
                neg_count = target.numel() - pos_count
                w0 = (neg_count + pos_count) / (2 * neg_count)
                w1 = (neg_count + pos_count) / (2 * pos_count)
                class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

                loss = F.binary_cross_entropy_with_logits(
                    output, target, pos_weight=class_weights[1]
                )
        else:
            loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)

        # Accuracy
        preds = torch.sigmoid(output).ge(0.5).float()
        acc = (preds == target).float().mean()
    else:
        raise ValueError(f"Unknown criterion: {cfg.criterion}")

    return loss, acc



def train_epoch(summary, summary_dev, cfg, args, model, dataloader,
                dataloader_dev, optimizer, summary_writer, best_dict,
                dev_header):
    torch.set_grad_enabled(True)
    model.train()
    device_ids = list(map(int, args.device_ids.split(',')))
    #device = torch.device('cuda:{}'.format(device_ids[0]))
    #print(f"device:{device}")
    steps = len(dataloader)
    dataiter = iter(dataloader)
    label_header = dataloader.dataset._label_header
    num_tasks = len(cfg.num_classes)
    #print(f"num_tasks:{num_tasks}")

    accumulation_steps = 8  # Set accumulation steps
    optimizer.zero_grad()  # Reset gradients accumulation
    accumulated_loss = 0.0  # For logging


    time_now = time.time()
    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    for step in tqdm(range(steps), desc=f"Epoch {summary['epoch'] + 1} [Train]"):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        #output, logit_map = model(image)
        # Convnext
        output, _ = model(image)
        #print(f"Parameter summary table: {SUMMARY(model, image)}")
        

        # different number of tasks
        loss = 0       
        loss, acc = get_loss(output, target, cfg)
        #print(f"ce_loss.shape: {loss_t.shape}")
        #loss += loss_t
        loss += loss / accumulation_steps
        loss_sum += loss.item()
        acc_sum += acc.item()

        #optimizer.zero_grad()
        #print(f"loss:{loss}")
        loss.backward()
        #optimizer.step()
        accumulated_loss += loss.item() * accumulation_steps
        if (step + 1) % accumulation_steps == 0 or (step + 1 == steps):
            optimizer.step()  # Update weights
            optimizer.zero_grad()  # Reset gradients


        summary['step'] += 1

        if summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg.log_every
            acc_sum /= cfg.log_every
            loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_sum))
            total_loss =  '{:.5f}'.format(loss.item())
            acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_sum))

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Total loss : {}, Loss : {}, '
                'Acc : {}, Run Time : {:.2f} sec'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['epoch'] + 1, summary['step'], total_loss, loss_str,
                        acc_str, time_spent))

            summary_writer.add_scalar(
                'train/loss_{}'.format(label_header[0]), loss_sum, 
                summary['step'])
            summary_writer.add_scalar(
                'train/acc_{}'.format(label_header[0]), acc_sum,
                summary['step'])

            loss_sum = np.zeros(num_tasks)
            acc_sum = np.zeros(num_tasks)

            # Log total loss
            summary_writer.add_scalar('train/total_loss', loss.item(), summary['step'])


        if summary['step'] % cfg.test_every == 0:
            time_now = time.time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev)
            time_spent = time.time() - time_now

            auc = 0
            y_pred = predlist
            y_true = true_list
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            summary_dev['auc'] = np.array([auc])


            loss_dev_str = '{:.5f}'.format(summary_dev['loss'])

            acc_dev_str = '{:.3f}'.format(summary_dev['acc'])

            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['auc']))

            logging.info(
                '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},'
                'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))

            summary_writer.add_scalar(
                'dev/loss_{}'.format(dev_header[0]), summary_dev['loss'], 
                summary['step'])
            summary_writer.add_scalar(
                'dev/acc_{}'.format(dev_header[0]), summary_dev['acc'],
                summary['step'])
            summary_writer.add_scalar(
                'dev/auc_{}'.format(dev_header[0]), summary_dev['auc'][0],
                summary['step'])
            


            

            summary_writer.add_scalar('dev/total_loss', loss.item(), summary['step'])

            save_best = False
            mean_acc = summary_dev['acc']
            if mean_acc >= best_dict['acc_dev_best']:
                best_dict['acc_dev_best'] = mean_acc
                if cfg.best_target == 'acc':
                    save_best = True

            mean_auc = summary_dev['auc']
            if mean_auc >= best_dict['auc_dev_best']:
                best_dict['auc_dev_best'] = mean_auc
                if cfg.best_target == 'auc':
                    save_best = True

            mean_loss = summary_dev['loss']
            if mean_loss <= best_dict['loss_dev_best']:
                best_dict['loss_dev_best'] = mean_loss
                if cfg.best_target == 'loss':
                    save_best = True

                
            mean_total_loss = summary_dev['total_loss']
            if mean_total_loss <= best_dict['loss_total_dev_best']:
                best_dict['loss_total_dev_best'] = mean_total_loss
                if cfg.best_target == 'total_loss':
                    save_best = True


            if save_best:
                torch.save(
                    {'epoch': summary['epoch'],
                     'step': summary['step'],
                     'acc_dev_best': best_dict['acc_dev_best'],
                     'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'loss_total_dev_best': best_dict['loss_total_dev_best'],
                     'state_dict': model.module.state_dict()},
                    os.path.join(args.save_path, 'best{}.ckpt'.format(
                        best_dict['best_idx']))
                )
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1
                logging.info(
                    '{}, Best, Step : {}, Loss : {}, Acc : {},Auc :{},'
                    'Best Auc : {:.3f}' .format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
                        acc_dev_str,
                        auc_dev_str,
                        float(best_dict['auc_dev_best'])))
        model.train()
        torch.set_grad_enabled(True)
    summary['epoch'] += 1

    return summary, best_dict


def test_epoch(summary, cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device_ids = list(map(int, args.device_ids.split(',')))
    #device = torch.device('cuda:{}'.format(device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    loss_sum = 0
    acc_sum = 0
    total_loss = 0
    predlist, true_list = [], []

    for step in tqdm(range(steps), desc="Evaluating [Dev]"):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        # Convnext
        output, _ = model(image)
        #output, logit_map = model(image)
        # different number of tasks
        
        loss, acc = get_loss(output, target, cfg)
        #print(f"loss_t:{loss_t}")
        # AUC
        y_pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
        y_true = target.view(-1).cpu().detach().numpy()
        if step == 0:
            predlist = y_pred
            true_list = y_true
        else:
            predlist = np.append(predlist, y_pred)
            true_list = np.append(true_list, y_true)
        print(len(predlist), len(true_list))


        total_loss = total_loss + loss
        loss_sum += loss.item()
        acc_sum += acc.item()

    summary['total_loss'] = total_loss
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    #for task in range(num_tasks):
     #   print(f"Task {task + 1} - Total number of zeros: {total_zeros_per_task[task]}, Total number of ones: {total_ones_per_task[task]}")


    return summary, predlist, true_list


def run(args):
    with open(args.cfg_path) as f:
        cfg = edict(json.load(f))
        if args.verbose is True:
            print(json.dumps(cfg, indent=4))

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if args.logtofile is True:
        logging.basicConfig(filename=args.save_path + '/log.txt',
                            filemode="w", level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            json.dump(cfg, f, indent=1)

    device_ids = list(map(int, args.device_ids.split(',')))
    #num_devices = torch.cuda.device_count()
    #if num_devices < len(device_ids):
    #    raise Exception(
    #        '#available gpu : {} < --device_ids : {}'
     #       .format(num_devices, len(device_ids)))
    #device = torch.device('cuda:{}'.format(device_ids[0]))

    #model = Classifier(cfg)
    model = VIT(cfg)
    #model = convnext_base(pretrained=False, in_22k=False, num_classes=4)
    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model.to(device), (3, h, w))
    #model = DataParallel(model, device_ids=device_ids).to(device).train()
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    if args.pre_train is not None:
        if os.path.exists(args.pre_train):
            ckpt = torch.load(args.pre_train, map_location=device)
            model.module.load_state_dict(ckpt, strict=False)
            #    print(name, module)

    optimizer = get_optimizer(model.parameters(), cfg)

    src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
    dst_folder = os.path.join(args.save_path, 'classification')
    rc, size = subprocess.getstatusoutput('du --max-depth=0 %s | cut -f1'
                                          % src_folder)
    if rc != 0:
        raise Exception('Copy folder error : {}'.format(rc))
    rc, err_msg = subprocess.getstatusoutput('cp -R %s %s' % (src_folder,
                                                              dst_folder))
    if rc != 0:
        raise Exception('copy folder error : {}'.format(err_msg))

    copyfile(cfg.train_csv, os.path.join(args.save_path, 'train.csv'))
    copyfile(cfg.dev_csv, os.path.join(args.save_path, 'dev.csv'))

    dataloader_train = DataLoader(
        ImageDataset_Mayo(cfg.train_csv, cfg, mode='train'),
        batch_size=cfg.train_batch_size, num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    dataloader_dev = DataLoader(
        ImageDataset_Mayo(cfg.dev_csv, cfg, mode='dev'),
        batch_size=cfg.dev_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    dev_header = dataloader_dev.dataset._label_header

    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    
    summary_writer = SummaryWriter(args.save_path)
    epoch_start = 0
    best_dict = {
        "acc_dev_best": 0.0,
        "auc_dev_best": 0.0,
        "loss_dev_best": float('inf'),
        "loss_total_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1}

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        #ckpt = torch.load(ckpt_path, map_location=device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'])
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
        best_dict['loss_total_dev_best'] = ckpt['loss_total_dev_best']
        best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        epoch_start = ckpt['epoch']

    for epoch in range(epoch_start, cfg.epoch):
        lr = lr_schedule(cfg.lr, cfg.lr_factor, summary_train['epoch'],
                         cfg.lr_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        summary_train, best_dict = train_epoch(
            summary_train, summary_dev, cfg, args, model,
            dataloader_train, dataloader_dev, optimizer,
            summary_writer, best_dict, dev_header)

        time_now = time.time()
        summary_dev, predlist, true_list = test_epoch(
            summary_dev, cfg, args, model, dataloader_dev)
        time_spent = time.time() - time_now

        auc = 0
        y_pred = predlist
        y_true = true_list
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        summary_dev['auc'] = np.array([auc])

        
        loss_dev_str = '{:.5f}'.format(summary_dev['loss'])

        acc_dev_str = '{:.3f}'.format(summary_dev['acc'])

        auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                    summary_dev['auc']))
      

        logging.info(
            '{}, Dev, Step : {}, Loss : {}, Acc : {}, Auc : {},'
            'Mean auc: {:.3f} ''Run Time : {:.2f} sec' .format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                summary_train['step'],
                loss_dev_str,
                acc_dev_str,
                auc_dev_str,
                summary_dev['auc'].mean(),
                time_spent))

        
        summary_writer.add_scalar(
            'dev/loss_{}'.format(dev_header[0]), summary_dev['loss'],
            summary_train['step'])
        summary_writer.add_scalar(
            'dev/acc_{}'.format(dev_header[0]), summary_dev['acc'],
            summary_train['step'])
        summary_writer.add_scalar(
            'dev/auc_{}'.format(dev_header[0]), summary_dev['auc'][0],
            summary_train['step'])
        


        save_best = False

        mean_acc = summary_dev['acc']
        if mean_acc >= best_dict['acc_dev_best']:
            best_dict['acc_dev_best'] = mean_acc
            if cfg.best_target == 'acc':
                save_best = True

        mean_auc = summary_dev['auc']
        if mean_auc >= best_dict['auc_dev_best']:
            best_dict['auc_dev_best'] = mean_auc
            if cfg.best_target == 'auc':
                save_best = True

        mean_loss = summary_dev['loss']
        if mean_loss <= best_dict['loss_dev_best']:
            best_dict['loss_dev_best'] = mean_loss
            if cfg.best_target == 'loss':
                save_best = True

        mean_total_loss = summary_dev['total_loss']
        if mean_total_loss <= best_dict['loss_total_dev_best']:
            best_dict['loss_total_dev_best'] = mean_total_loss
            if cfg.best_target == 'total_loss':
                save_best = True

        if save_best:
            torch.save(
                {'epoch': summary_train['epoch'],
                 'step': summary_train['step'],
                 'acc_dev_best': best_dict['acc_dev_best'],
                 'auc_dev_best': best_dict['auc_dev_best'],
                 'loss_dev_best': best_dict['loss_dev_best'],
                 'loss_total_dev_best': best_dict['loss_total_dev_best'],
                 'state_dict': model.module.state_dict()},
                os.path.join(args.save_path,
                             'best{}.ckpt'.format(best_dict['best_idx']))
            )
            best_dict['best_idx'] += 1
            if best_dict['best_idx'] > cfg.save_top_k:
                best_dict['best_idx'] = 1
            logging.info(
                '{}, Best, Step : {}, Loss : {}, Acc : {},'
                'Auc :{},Best Auc : {:.3f}' .format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary_train['step'],
                    loss_dev_str,
                    acc_dev_str,
                    auc_dev_str,
                    float(best_dict['auc_dev_best'])))
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'acc_dev_best': best_dict['acc_dev_best'],
                    'auc_dev_best': best_dict['auc_dev_best'],
                    'loss_dev_best': best_dict['loss_dev_best'],
                    'loss_total_dev_best': best_dict['loss_total_dev_best'],
                    'state_dict': model.module.state_dict()},
                   os.path.join(args.save_path, 'train.ckpt'))
    summary_writer.close()


def main():
    args = parser.parse_args()
    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    run(args)


if __name__ == '__main__':
    main()
