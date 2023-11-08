
# imports
import argparse
import os
import glob
import numpy as np
import torch
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

home_path = os.path.dirname(os.getcwd())
import sys
sys.path.append(os.path.join(home_path,"optical-flow/src"))
sys.path.append(os.path.join(home_path,"focus-of-expansion"))
sys.path.append(os.path.join(home_path,"time-to-contact"))
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from foe import RANSAC
from ttc import get_ttc

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
            path = "example_frames/driving/"
        elif args.scene == "intersection":
            path = "example_frames/intersection/"
        
        images = os.listdir(path)
        for image in images:
            if not(image.endswith(".png")) and not(image.endswith(".jpg")):
                images.remove(image)
        images = sorted(images)
        images = [path + image for image in images]

        if len(images) < 3:
            print("Error: Expected at least 3 images")
            return

        # Load three images based on ordering in name (chronological by processed KITTI)
        for i in range(len(images) - 2):
            imfile1 = images[i]
            imfile2 = images[i + 1]
            imfile3 = images[i + 2]

            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            image3 = load_image(imfile3)

            padder = InputPadder(image1.shape)
            image1, image2, image3 = padder.pad(image1, image2, image3)

            # get optical flow
            coords0, coords1, flow_up1 = model(image1, image2, iters=20, test_mode=True)
            coords2, coords3, flow_up2 = model(image2, image3, iters=20, test_mode=True)
            optical_flow1 = flow_up1[0].permute(1,2,0).cpu().numpy()
            optical_flow2 = flow_up2[0].permute(1,2,0).cpu().numpy()
        
            grad_flow = optical_flow2 - optical_flow1

            gf_magn = np.sqrt(np.sum(np.power(grad_flow,2),axis=2))
            gf_magn = gf_magn / gf_magn.max()
            
            gf_dir = (np.arctan2(grad_flow[:,:,0],grad_flow[:,:,1]) + np.pi) / (2*np.pi)
            
            dir_map = plt.get_cmap('hsv')
            gf_magn_GS = cv2.cvtColor(gf_magn,cv2.COLOR_GRAY2RGB)
            gf_dir_RGB = np.multiply(dir_map(gf_dir)[:,:,:3],255.0)
            gf_overlap_RGB = np.multiply(np.divide(gf_magn_GS,255.0),gf_dir_RGB)
            
            # Perform 2 component ICA on MAG and DIR separately
            ica_mag = FastICA(n_components=2)
            ica_dir = FastICA(n_components=2)
            
            gMag_ica = ica_mag.fit_transform(gf_magn)
            gDir_ica = ica_dir.fit_transform(gf_dir)

            gMag_rec = ica_mag.inverse_transform(gMag_ica)
            gDir_rec = ica_dir.inverse_transform(gDir_ica)

            # OF 2-Piece ICA by running FastICA(x) and FastICA(y)
            # Working with the assumption that we can separate the signal into two distinct linear signals
            
            # ICA maintains signal variance, but it fails to preserve spatial relationships

            ica_OF_x = FastICA(n_components=1)
            ica_OF_y = FastICA(n_components=1)

            OF_shape = optical_flow1.shape[:2]
            OF_x = optical_flow1[:,:,0]
            OF_y = optical_flow1[:,:,1]

            x_rec = ica_OF_x.fit_transform(OF_x)
            y_rec = ica_OF_y.fit_transform(OF_y)

            x_inv = ica_OF_x.inverse_transform(x_rec)
            x_inv_vis = cv2.cvtColor(np.abs(x_inv)/np.abs(x_inv).max(),cv2.COLOR_GRAY2RGB)

            y_inv = ica_OF_y.inverse_transform(y_rec)
            y_inv_vis = cv2.cvtColor(np.abs(y_inv)/np.abs(y_inv).max(),cv2.COLOR_GRAY2RGB)

            fig, ((pl1, pl3), (pl2, pl4)) = plt.subplots(2,2)
            pl1.imshow(cv2.cvtColor(np.abs(OF_x)/np.abs(OF_x).max(),cv2.COLOR_GRAY2RGB))
            pl2.imshow(x_inv_vis)
            pl3.imshow(cv2.cvtColor(np.abs(OF_y)/np.abs(OF_y).max(),cv2.COLOR_GRAY2RGB))
            pl4.imshow(y_inv_vis)
            plt.show()

            continue

            fig, ((pl1, pl4), (pl2, pl5), (pl3, pl6)) = plt.subplots(3,2)
            x_rec_1 = np.reshape(x_rec[:,0],(-1,1))
            x_rec_2 = np.reshape(x_rec[:,1],(-1,1))
            pl1.imshow(x_rec_1 @ np.reshape(ica_OF_x.components_[0],(1,-1)) + x_rec_1 @ np.reshape(ica_OF_x.components_[1],(1,-1)))
            pl2.imshow(x_rec_2 @ np.reshape(ica_OF_x.components_[0],(1,-1)) + x_rec_2 @ np.reshape(ica_OF_x.components_[1],(1,-1)))
            pl3.imshow(np.abs(ica_OF_x.inverse_transform(x_rec)))
            
            y_rec_1 = np.reshape(y_rec[:,0],(-1,1))
            y_rec_2 = np.reshape(y_rec[:,1],(-1,1))
            pl4.imshow(y_rec_1 @ np.reshape(ica_OF_y.components_[0],(1,-1)) + y_rec_1 @ np.reshape(ica_OF_y.components_[1],(1,-1)))
            pl5.imshow(y_rec_2 @ np.reshape(ica_OF_y.components_[0],(1,-1)) + y_rec_2 @ np.reshape(ica_OF_y.components_[1],(1,-1)))
            pl6.imshow(np.abs(ica_OF_y.inverse_transform(y_rec)))

            #fig2, pl2_1 = plt.subplots(1,1)
            #pl2_1.imshow(ica_OF_x.inverse_transform(x_rec) + ica_OF_y.inverse_transform(y_rec))
            plt.show()

            continue

            OF_magn = np.sqrt(np.sum(np.power(optical_flow1,2),axis=2))
            OF_magn = OF_magn / OF_magn.max()
            OF_dir = (np.arctan2(grad_flow[:,:,0],grad_flow[:,:,1]) + np.pi) / (2 * np.pi)

            #OF_combo = 255 * (((OF_magn * OF_dir) + np.pi) / (2*np.pi))
            OF_combo = 25.5 * (np.floor((10*OF_magn)) + OF_dir)
            plt.imshow(cv2.cvtColor(OF_combo,cv2.COLOR_GRAY2RGB).astype('int'))
            plt.savefig(args.scene + "_of_combo_optical_flow.png")

            OF_ica = ica.fit_transform(OF_combo)
            
            OF_ica_inv = ica.inverse_transform(OF_ica)
            #OF_ica_inv = 255.0/np.max(OF_ica_inv)
            OF_iso = OF_magn * 255 - OF_ica_inv

            plt.imshow(cv2.cvtColor(OF_iso,cv2.COLOR_GRAY2RGB).astype('int'))
            plt.savefig(args.scene + "_inv_iso_optical_flow.png")

            plt.imshow(cv2.cvtColor(OF_ica_inv,cv2.COLOR_GRAY2RGB).astype('int'))
            plt.savefig(args.scene + "_inv_ica_optical_flow.png")

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
