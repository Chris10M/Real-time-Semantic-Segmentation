import time
import torch
from torch.cuda.amp import autocast
from torchstat import stat

from cityscapes import CityScapes
from model.model import get_network


def get_flops(net, input_size):
    print(stat(net, (3, *input_size)))


def calculate_fps(net, input_size, steps=1000):
    inp = torch.rand((1, 3, *input_size)).cuda()
    net.eval()

    inference_time = 0
    for i in range(0, steps):
        start_time = time.time()
        
        # with autocast():
        with torch.no_grad():
                    out = net(inp)
                
        end_time = time.time() - start_time

        inference_time +=end_time

    inference_time /= steps

    return 1 / inference_time


def main():
    scale = 1
    cropsize = [int(2048 * scale), int(1024 * scale)]
    
    net = get_network(19)
    net.eval()
    
    fps = calculate_fps(net, cropsize)
    get_flops(net.cpu(), cropsize)

    print(f'fps: {fps}')

if __name__ == '__main__':
    main()