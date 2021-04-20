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

import sys
import torch
import collections
import torch.nn.functional as F
import numpy as np
import cv2
import os
import utils
from model import model
from arg_parser import evaluate

from tabulate import tabulate
from sklearn import metrics
from torch.utils.data import DataLoader
from cityscapes import CityScapes
from tqdm import tqdm

from arg_parser import evaluate


RESULTS_ROOT = 'cityscapes_results'
os.makedirs(RESULTS_ROOT, exist_ok=True)


class Evaluate:
    def __init__(self, dataset, net):
        self.net = net
        self.ds = dataset

        self.n_classes = dataset.n_classes

    def __call__(self, imgpths, im, lb):
        self.net.eval()

        with torch.no_grad():
            out = self.net(im)

            preds = out.argmax(dim=1).cpu().numpy()

        for img_path, pred in zip(imgpths, preds):
            pred = self.ds.convert_labels_to_ids(pred)
            
            file_name = img_path.split('/')[-1]
            save_path = os.path.join(RESULTS_ROOT, file_name)
            
            cv2.imwrite(save_path, pred)
        
        self.net.train()

    def compute(self):
        os.environ['CITYSCAPES_DATASET'] = self.ds.rootpth
        os.environ['CITYSCAPES_RESULTS'] = RESULTS_ROOT
        sys.argv = sys.argv[:1]
        
        from cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling
        evalPixelLevelSemanticLabeling.main()
        
        return 

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
    
    for imgpths, im, lb in tqdm(dl):
        with torch.no_grad():
            if torch.cuda.is_available(): 
                im = im.cuda()
    
            evaluate(imgpths, im, lb)
        
    evaluate.compute()


if __name__ == "__main__":
    args = evaluate()
    main(args)
