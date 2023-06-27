
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
sys.path.append("/home/kristinchengwu/dev/video-segmentation/focus-of-expansion")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from foe import RANSAC

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_foe(coords0, coords1):
    # convert to numpy 
    coords0 = coords0[0].permute(1, 2, 0).cpu().numpy()
    coords1 = coords1[0].permute(1, 2, 0).cpu().numpy()

    coords0 = coords0.reshape((coords0.shape[0]*coords0.shape[1], 2))
    coords1 = coords1.reshape((coords1.shape[0]*coords1.shape[1], 2))
    coords = np.hstack((coords0, coords1))
    return coords

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def viz(img, flo, foe):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    foe_x = foe[0]*8
    foe_y = foe[1]*8

    plt.plot(foe_x, foe_y, marker="v", color="red")
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
            coords = get_foe(coords0, coords1)
            foe, inlier_ratio, inliers = RANSAC(coords)

            # visualization
            viz(image1, flow_up, foe)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    main(args)