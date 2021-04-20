"""
The train.py needs two arguments,
--root     (compulsary) - root directory of Cityscapes 
--model_path (optional) - model_path to resume training from a checkpoint

The trained model is evaluated and saved for every save_iter. The logs and model configuration are saved in savedmodels/
with the hash of the model as the base_folder. This helps us to track different models. 

We only optimize the loss for every optim_iter, this mimics increase in the batchsize. 
"""

from torch.cuda.amp import autocast, GradScaler
from logger import setup_logger
import torch.multiprocessing as mp
from model import model
import os
from cityscapes import CityScapes
from optimizer import Optimzer
from loss import OhemCELoss, IoULoss, OHIoULoss, DiceLoss
from evaluate import evaluate_net
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist
import hashlib
import os
import os.path as osp
import logging
import time
import datetime
import argparse
import imutils

from arg_parser import train


class Logger:
    logger = None
    ModelSavePath = 'model'

def set_model_logger(net):
    model_info = str(net)

    respth = f'savedmodels/{hashlib.md5(model_info.encode()).hexdigest()}'
    Logger.ModelSavePath = respth

    if not osp.exists(respth): os.makedirs(respth)
    logger = logging.getLogger()

    if setup_logger(respth):
        logger.info(model_info)

    Logger.logger = logger


def main(args):
    cropsize = [768, 768]

    cityscapes_path = args.cityscapes_path
    ds = CityScapes(cityscapes_path, cropsize=cropsize, mode='train')

    n_classes = ds.n_classes    
    net = model.get_network(n_classes)

    set_model_logger(net)
    
    saved_path = args.saved_model
    
    max_iter = 64000
    optim_iter = 64 
    save_iter = 1000
    n_img_per_gpu = 6
    n_workers = min(n_img_per_gpu, 16)
    
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=True,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)

    ## model
    ignore_idx = 255

    score_thres = 0.7
    criteria = OhemCELoss(thresh=score_thres, ignore_lb=ignore_idx)
    if torch.cuda.is_available: criteria = criteria.cuda()

    optim = Optimzer(net, 0, max_iter)

    min_eval_loss = 1e5
    epoch = 0
    start_it = 0
    if os.path.isfile(saved_path):
        loaded_model = torch.load(saved_path)
        state_dict = loaded_model['state_dict']

        try:
            net.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(e)
        
        try:
            start_it = 0
            start_it = loaded_model['start_it'] + 2
        except KeyError:
            start_it = 0

        try:
            epoch = loaded_model['epoch']
        except KeyError:
            epoch = 0

        try:
            min_eval_loss = loaded_model['min_eval_loss']
        except KeyError: ...

        try:
            optim = Optimzer(net, start_it, max_iter)
            optim.load_state_dict(loaded_model['optimize_state'])
            ...
        except (ValueError, KeyError): pass


        print(f'Model Loaded: {saved_path} @ start_it: {start_it}')

    ## train loop
    msg_iter = 50
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)

    start_training = False
    for it in range(start_it, max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            diter = iter(dl)
            im, lb = next(diter)

        im = im.cuda()
        lb = lb.cuda()

        if not start_training:
            start_training = True

        outs = net(im)
        if isinstance(outs, tuple):  # Depending on the model, AuxLoss may be also computed.
            loss = criteria(outs[0], lb)
            for out in outs[1:]:
                loss += criteria(out, lb)
        else:
            out = outs
            loss = criteria(out, lb)

        loss /= optim_iter

        loss.backward()

        if it % optim_iter == 0:  # Gradient accumulation.
            optim.update_lr()

            optim.step()
            optim.zero_grad()

        loss_avg.append(loss.item())

        if (it + 1) % save_iter == 0 or os.path.isfile('save'):
            save_pth = osp.join(Logger.ModelSavePath, f'{it + 1}_{int(time.time())}.pth')

            evaluation = evaluate_net(args, net)
            Logger.logger.info(f"Model@{it + 1}\n{evaluation}")
            
            eval_loss = evaluation.loss()
            optim.reduce_lr_on_plateau(eval_loss)

            if eval_loss < min_eval_loss:  
                print(f'Saving model at: {(it + 1)}, save_pth: {save_pth}')
                torch.save({
                    'epoch': epoch,
                    'start_it': it,
                    'state_dict': net.state_dict(),
                    'optimize_state': optim.state_dict(),
                    'min_eval_loss': min_eval_loss,
                }, save_pth)
                print(f'model at: {(it + 1)} Saved')

                min_eval_loss = eval_loss

        #   print training log message
        if (it+1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    f'epoch: {epoch}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            Logger.logger.info(msg)
            loss_avg = []
            st = ed

    save_pth = osp.join(Logger.ModelSavePath, 'model_final.pth')
    net.cpu()
    torch.save({'state_dict': net.state_dict()}, save_pth)

    Logger.logger.info('training done, model saved to: {}'.format(save_pth))


if __name__ == "__main__":
    args = train()
    main(args)
