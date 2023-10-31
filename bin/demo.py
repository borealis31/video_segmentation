
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
            gf_magn = gf_magn * 255 / gf_magn.max()
            
            
            
            gf_magn_rgb = cv2.cvtColor(gf_magn,cv2.COLOR_GRAY2RGB)
            
            dir_map = plt.get_cmap('hsv')
            gf_dir = (np.arctan2(grad_flow[:,:,0],grad_flow[:,:,1]) + np.pi) / (2*np.pi)
            gf_dir = np.multiply(dir_map(gf_dir)[:,:,:3],255.0)

            gf_overlap = np.multiply(np.divide(gf_magn_rgb,255.0),gf_dir)
            
            # Linear ICA, probably not going to be super useful for this project but provides some starting point
            ica = FastICA(n_components=3)
            ica.fit(gf_magn)
            gf_ica = ica.fit_transform(gf_magn)
            gf_ica_inv = ica.inverse_transform(gf_ica)
            gf_ica_inv = 255.0*gf_ica_inv/np.max(gf_ica_inv)
            gf_iso = gf_magn - gf_ica_inv

            plt.imshow(cv2.cvtColor(gf_iso,cv2.COLOR_GRAY2RGB).astype('int'))
            plt.savefig(args.scene + "_inv_iso.png")

            plt.imshow(cv2.cvtColor(gf_ica_inv,cv2.COLOR_GRAY2RGB).astype('int'))
            plt.savefig(args.scene + "_inv_ica.png")

            flo1 = flow_viz.flow_to_image(optical_flow1)
            flo2 = flow_viz.flow_to_image(optical_flow2)
            
            vis_img1 = image1
            vis_img1 = vis_img1[0].permute(1,2,0).cpu().numpy()
            vis_img2 = image2
            vis_img2 = vis_img2[0].permute(1,2,0).cpu().numpy()
            vis_img3 = image3
            vis_img3 = vis_img3[0].permute(1,2,0).cpu().numpy()

            fig, (pl1, pl2) = plt.subplots(2,1)
            fig.subplots_adjust(top=0.93, bottom=0.15, hspace=0.01)
            pl1.imshow(vis_img2.astype('int'))
            pl2.imshow(gf_magn_rgb.astype('int'))
            fig.suptitle("Optical Flow Gradient Magnitude (Normalized)")
            fig.savefig(args.scene + "_grad_magn.png")

            pl1.cla()
            pl2.cla()

            pl3 = fig.add_axes([0.15, 0.1, 0.7, 0.025])
            pl1.imshow(vis_img2.astype('int'))
            pl2.imshow(gf_dir.astype('int'))
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=360), cmap=mpl.cm.hsv),
                         cax=pl3, orientation="horizontal",label="Gradient Direction [deg]")
            fig.suptitle("Optical Flow Direction")
            fig.savefig(args.scene + "_grad_dir.png")

            pl1.cla()
            pl2.cla()

            pl1.imshow(vis_img2.astype('int'))
            pl2.imshow(gf_overlap.astype('int'))
            fig.suptitle("Optical Flow Magnitude and Direction Overlayed")
            fig.savefig(args.scene + "_grad_overlap.png")
            plt.clf()

            # Let's just look at OF and see how that responds to FastICA
            # NOTE: FastICA does not support 3-dim data, we are limited to 2D
            # As a means of combining direction and magnitude, we can round normalized
            # magnitude into a [0,255] interval and attach direction as a decimal part
            # with normalized radians from [0, 1]
            ica_OF = FastICA(n_components=3)
            
            OF_magn = np.sqrt(np.sum(np.power(optical_flow1,2),axis=2))
            OF_magn = OF_magn / OF_magn.max()
            OF_dir = (np.arctan2(grad_flow[:,:,0],grad_flow[:,:,1]) + np.pi) / (2 * np.pi)

            #OF_combo = 255 * (((OF_magn * OF_dir) + np.pi) / (2*np.pi))
            OF_combo = 25.5 * (np.floor((10*OF_dir)) + OF_magn)
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
