
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
            img = image1
            flo = flow_up
            img = img[0].permute(1,2,0).cpu().numpy()

            outliers = set(range(0, len(coords))) - set(inliers)
            for outlier in outliers:
                outlier = round(outlier)
                i1 = round(coords[outlier][1])*8
                i2 = round(coords[outlier][0])*8
                img[i1:(i1+8), i2:(i2+8) ] =  np.array([255, 0, 0])

            flo = flo[0].permute(1,2,0).cpu().numpy()
            flo = flow_viz.flow_to_image(flo)

            img_flo = np.concatenate([img, flo], axis=0)
            plt.plot(foe[0]*8, foe[1]*8, marker="v", color="white")
            plt.imshow(img_flo / 255.0)
            plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    main(args)