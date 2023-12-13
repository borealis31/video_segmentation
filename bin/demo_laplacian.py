
# imports
import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

home_path = os.path.dirname(os.getcwd())
import sys
sys.path.append(os.path.join(home_path,"optical-flow/src"))
sys.path.append(os.path.join(home_path,"focus-of-expansion"))
sys.path.append(os.path.join(home_path,"time-to-contact"))
sys.path.append(home_path)
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from foe import RANSAC
from ttc import get_ttc
from pyiva.iva_laplace import *
from sklearn.decomposition import FastICA

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        # Get images from image directory (default example_frames)
        if args.scene == "driving":
            path = "example_frames/data/raw_driving/"
        elif args.scene == "intersection":
            path = "example_frames/data/raw_intersection/"
        
        # Extract string names of images and sort by sequence image number
        images = os.listdir(path)
        for image in images:
            if not(image.endswith(".png")) and not(image.endswith(".jpg")):
                images.remove(image)
        images = sorted(images)
        images = [path + image for image in images]

        if len(images) < 2:
            print("Error: Expected at least 2 images")
            return

        fig, pl1 = plt.subplots(1,1)
        anim_set = []
        # Load successive images based on ordering in name (chronological by processed KITTI)
        j = len(images) - 1 if len(images) - 1 < 10 else 10
        for i in range(j):
            print("Starting OF {} of {}".format(i+1,j))
            imfile1 = images[i]
            imfile2 = images[i + 1]

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # Obtain optical flow through RAFT
            coords0, coords1, flow_up1 = model(image1, image2, iters=20, test_mode=True)
            optical_flow1 = flow_up1[0].permute(1,2,0).cpu().numpy()
            
            ## LAPLACIAN
            # Idea: Employ the Laplacian Operator on the OF to try to eliminate ego-motion
            OF_lap_x = cv2.Laplacian(optical_flow1[:,:,0],ddepth=-1)
            OF_lap_x /= OF_lap_x.max()
            OF_lap_y = cv2.Laplacian(optical_flow1[:,:,1],ddepth=-1)
            OF_lap_y /= OF_lap_y.max()

            # Theshold the OF crossing strength to 1 for masking
            OF_thresh = np.float32(np.array([[1 if (np.abs(OF_lap_x[l,k]) > 0.3 or np.abs(OF_lap_y[l,k]) > 0.3) else 0 for k in range(OF_lap_x.shape[1])] for l in range(OF_lap_x.shape[0])]))
            anim_im = pl1.imshow(cv2.cvtColor(OF_thresh,cv2.COLOR_GRAY2RGB), animated=True)
            if i == 0:
                pl1.imshow(cv2.cvtColor(OF_thresh,cv2.COLOR_GRAY2RGB))
            anim_set.append([anim_im])
        
        # Create animation of output
        anim = animation.ArtistAnimation(fig,anim_set,interval=50,blit=True,repeat_delay=1000)
        writergif = animation.PillowWriter(fps=1)
        anim.save('laplacian_sequence.gif',writer=writergif)
        
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="../optical-flow/models/raft-things.pth",help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--sample', default="16",help="use a square of sample to analyze FoEs")
    parser.add_argument('--scene', default="driving", help="select scene for analysis (must be driving or intersection)")
    args = parser.parse_args()

    main(args)
