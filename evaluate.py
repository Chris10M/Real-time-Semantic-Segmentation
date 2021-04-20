"""
The evaluate.py needs two arguments,
--root       (compulsary) - root directory of Cityscapes 
--model_path (compulsary) - path of the saved_model  


The trained model is evaluated on Cityscapes validation dataset. The metrics we calculate are,
accuracy, f1-score, sensitivity, jaccardSimilarity, diceScore, IoU.

We use the gt with label 255 as negative and 0-19 are postivies for all metrics except IoU. FN - (gt == 255) and (pred in (0-19)). 
While for Iou, we ignore the 255, and calculate with 0-19 classes only. This is done to ensure the results can be compared with other 
networks for Cityscapes.   
"""

import torch
import collections
import torch.nn.functional as F
import numpy as np
import cv2
from model import model

from tabulate import tabulate
from sklearn import metrics
from torch.utils.data import DataLoader
from cityscapes import CityScapes
from tqdm import tqdm

from arg_parser import evaluate


class Evaluate:
    def __init__(self, dataset, net):
        self.net = net
        self.ds = dataset

        self.n_classes = 19

        self.steps = 0
        self.metrics = collections.defaultdict(dict)
        
        self.class_metrics = dict()
        self.macro_metrics = dict()

        self.ignore_lb = torch.tensor(self.ds.ignore_lb, dtype=torch.int64).cuda() if torch.cuda.is_available() else torch.tensor(self.ds.ignore_lb, dtype=torch.int64)

    def __call__(self, im, lb):
        self.net.eval()

        with torch.no_grad():
            out = self.net(im)

            preds = out.argmax(dim=1)
            
        lb_flat = lb.view(-1)
        preds_flat = preds.view(-1)

        self.append_class_wise(lb_flat, preds_flat)
        self.steps += 1

        self.net.train()

    def append_class_wise(self, lb_flat, preds_flat):
        gto = (lb_flat == self.ignore_lb)
        gtn = (lb_flat != self.ignore_lb)
        
        for class_id in range(0, self.n_classes):
            gt = (lb_flat == class_id)
            pred = (preds_flat == class_id)

            eq = torch.logical_and(gt, pred)
            ne = torch.not_equal(gt, pred)

            tp = int(torch.count_nonzero(torch.logical_and(eq, gtn)))
            tn = int(torch.count_nonzero(torch.logical_and(eq, gto)))
            
            fp = int(torch.count_nonzero(torch.logical_and(ne, gtn)))
            fn = int(torch.count_nonzero(torch.logical_and(ne, gto)))
            
            try: self.metrics[class_id]['tn'] += tn
            except KeyError: self.metrics[class_id]['tn'] = tn

            try: self.metrics[class_id]['fp'] += fp
            except KeyError: self.metrics[class_id]['fp'] = fp

            try: self.metrics[class_id]['fn'] += fn
            except KeyError: self.metrics[class_id]['fn'] = fn

            try: self.metrics[class_id]['tp'] += tp
            except KeyError: self.metrics[class_id]['tp'] = tp

    def compute_metrics(self):
        macro_metrics = {
            'IoU': 0
        }
        class_metrics = dict()
        for class_id in range(0, self.n_classes):  
            tp = self.metrics[class_id]['tp']
            fp = self.metrics[class_id]['fp']
            fn = self.metrics[class_id]['fn']
            tn = self.metrics[class_id]['tn']

            try: iou = tp / (tp + fp) 
            except ZeroDivisionError: iou = 0

            class_info = self.ds.get_class_info(class_id)

            class_metrics[class_info['name']] = {
                'IoU': iou
            }

            macro_metrics['IoU'] += iou  

        macro_metrics['IoU'] /= self.n_classes  

        self.macro_metrics = macro_metrics
        self.class_metrics = class_metrics

    def __str__(self):
        self.compute_metrics()

        macro_table = [["mean", *[round(f, 3) for f in self.macro_metrics.values()]]]
        macro_table = tabulate(macro_table, headers=self.macro_metrics.keys(), tablefmt="pretty")
    
        micro_table = [[k, *[round(f, 3) for f in v.values()]] for k, v in self.class_metrics.items()]
        micro_table = tabulate(micro_table, headers=self.macro_metrics.keys(), tablefmt="pretty")

        table = f'{macro_table}\n\n{micro_table}'
    
        return table

    def loss(self):
        mean = list()
        for f in self.macro_metrics.values():
            mean.append(f)

        return 1 - (sum(mean) / len(mean))


def evaluate_net(args, net):
    scale = 1
    cropsize = [int(2048 * scale), int(1024 * scale)]

    cityscapes_path = args.cityscapes_path

    ds = CityScapes(cityscapes_path, cropsize=cropsize, mode='val')

    dl = DataLoader(ds,
                    batch_size=8,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True)

    evaluate = Evaluate(ds, net)

    print('Evaluate model')
    for impth, im, lb in tqdm(dl):
        with torch.no_grad():
            if torch.cuda.is_available():
                im = im.cuda()
                lb = lb.cuda()
                
            evaluate(im, lb)

    return evaluate


def main(args):
    scale = 1
    cropsize = [int(2048 * scale), int(1024 * scale)]

    cityscapes_path = args.cityscapes_path

    ds = CityScapes(cityscapes_path, cropsize=cropsize, mode='val')
    n_classes = ds.n_classes
    
    dl = DataLoader(ds,
                    batch_size=10,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=True)

    net = model.get_network(n_classes)

    saved_path = args.saved_model
    
    print(saved_path)

    loaded_model = torch.load(saved_path, map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
    state_dict = loaded_model['state_dict']
    net.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        net.cuda()

    net.eval()
    evaluate = Evaluate(ds, net)
    
    for images, im, lb in tqdm(dl):
        with torch.no_grad():
            if torch.cuda.is_available(): 
                im = im.cuda()
                lb = lb.cuda()

            evaluate(im, lb)
                
    print(evaluate)


if __name__ == "__main__":
    args = evaluate()
    main(args)
