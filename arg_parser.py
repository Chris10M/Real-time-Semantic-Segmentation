import argparse

def demo():
    parser = argparse.ArgumentParser(description='Demo for Cityscapes')
    
    parser.add_argument('--root', dest='cityscapes_path', required=True,
                        help='root path of Cityscapes')

    parser.add_argument('--model_path', dest='saved_model', required=True,
                        help='path of saved_model')
    args = parser.parse_args()

    return args


def train():
    parser = argparse.ArgumentParser(description='Trainer for Cityscapes')
    
    parser.add_argument('--root', dest='cityscapes_path', required=True,
                        help='root path of Cityscapes')

    parser.add_argument('--model_path', dest='saved_model', required=False,
                        help='path of saved_model', default="")
    args = parser.parse_args()

    return args


def evaluate():
    parser = argparse.ArgumentParser(description='evaluation of Cityscapes')
    
    parser.add_argument('--root', dest='cityscapes_path', required=True,
                        help='root path of Cityscapes')

    parser.add_argument('--model_path', dest='saved_model', required=True,
                        help='path of saved_model')
    args = parser.parse_args()

    return args


def demo_single():
    parser = argparse.ArgumentParser(description='Demo for Cityscapes')
    
    parser.add_argument('--model_path', dest='saved_model', required=True,
                        help='path of saved_model')

    parser.add_argument('--img_path', dest='img_path', required=False,
                        help='path of image to infer', default="images/demo.png")

    args = parser.parse_args()

    return args
