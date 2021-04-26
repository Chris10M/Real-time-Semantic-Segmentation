"""
To visualize the results, demo.py needs two arguments,
--model_path (compulsary) - path of the saved_model  
--img_path (optional) - image to evaluate, default takes, "images/demo.png" 

Press 'q' to quit the demo.
Press any key to visualize the next image.   
"""


import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import imutils
from PIL import Image

from torch.utils.data import DataLoader
from cityscapes import CityScapes
from model import model
from arg_parser import demo_single


def to_tensor(img):
    return transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            ])(img)


def main(args):
    scale = 1
    cropsize = [int(2048 * scale), int(1024 * scale)]

    img_path = args.img_path

    ds = CityScapes("", cropsize=cropsize, mode='val', demo=True)
    n_classes = ds.n_classes

    net = model.get_network(n_classes)

    saved_path = args.saved_model
    loaded_model = torch.load(saved_path, map_location=torch.device('cuda') if torch.cuda.is_available() else 'cpu')

    state_dict = loaded_model['state_dict']

    net.load_state_dict(state_dict, strict=False)

    if torch.cuda.is_available():
        net.cuda()

    net.eval()

    img = Image.open(img_path).convert('RGB')
    im = to_tensor(img).unsqueeze(0)
    
    with torch.no_grad():
        if torch.cuda.is_available():
            im = im.cuda()

        pred = net(im).argmax(dim=1).squeeze(0).cpu().numpy()
        
        pred = ds.vis_label(pred)
        image = np.array(img)[:, :, ::-1]

        cv2.imshow('demo', imutils.resize(cv2.hconcat([image, pred]), width=1920))
        cv2.waitKey(0)


if __name__ == '__main__':
    args = demo_single()
    main(args)
