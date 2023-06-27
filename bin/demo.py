
# imports
import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/kristinchengwu/dev/video-segmentation/optical-flow/src")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    plt.imshow(img_flo / 255.0)
    plt.show()

def main(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
    model = model.module
    model.to(device)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)

        for imfile1, imfile2 in zip(images[:-1], images[1:]):

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            coords0, coords1, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    main(args)