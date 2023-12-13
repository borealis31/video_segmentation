
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
sys.path.append(home_path)
sys.path.append(os.path.join(home_path,"VAE-BSS"))
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

            # Calculate Optical Flow of 1->2 and 2->3
            coords0, coords1, flow_up1 = model(image1, image2, iters=20, test_mode=True)
            coords2, coords3, flow_up2 = model(image2, image3, iters=20, test_mode=True)
            
            optical_flow1 = flow_up1[0].permute(1,2,0).cpu().numpy()
            optical_flow2 = flow_up2[0].permute(1,2,0).cpu().numpy()
        
            # Calculate Gradient of OF for this image sequence
            grad_flow = optical_flow2 - optical_flow1

            gf_magn = np.sqrt(np.sum(np.power(grad_flow,2),axis=2))
            gf_magn = gf_magn / gf_magn.max()
            
            ## BLURRED COMBINATION
            # Idea: A Gaussian Blurred image of the OF field in addition to the OF, we may be able to recover:
            # 1. Egomotion Field (Approx. Same in OF + Blurred)
            # 2. Motion from OF
            # NOTE: FastICA/pyiva require that the number of components is greater than or equal to the number of extractable
            #       sources, e.g. we cannot extract d sources from an NxF matrix when d > N
            # Results: Gaussian blur identified as the second source, not viable
            ##

            """
            ## SUCCESSIVE + GRADIENT COMBINATION
            # Idea: Using Successive OF fields + the Gradient field, we can add a third source that may enable us to recover:
            # 1. Egomotion Field (Same in successive OFs + approx. 0 in Gradient)
            # 2. Motion from OF 1
            # 3. Motion from OF 2
            # Results: We get a few interesting resultant images, looks like we get an estimate of ego-motion, but it includes the center of the vehicle

            # Perform IVA using pyiva
            OF_shape = optical_flow1[:,:,0].shape
            print(OF_shape)
            C1 = np.ravel(grad_flow[:,:,0])
            C2 = np.ravel(optical_flow1[:,:,0])
            C3 = np.ravel(optical_flow2[:,:,0])
            COMB_IVA = np.array([[C1, C2, C3]])
            COMB_IVA /= np.abs(COMB_IVA).max()
            W_est = iva_laplace(COMB_IVA)
            S_IVA = W_est[0] @ COMB_IVA[0]
            S1 = np.reshape(S_IVA[0,:],OF_shape)
            S2 = np.reshape(S_IVA[1,:],OF_shape)
            S3 = np.reshape(S_IVA[2,:],OF_shape)

            # Perform linear ICA using FastICA
            ica_COMB_x = FastICA(n_components=3,max_iter=500)
            S_F = ica_COMB_x.fit_transform(COMB_IVA[0,:,:].T)
            S1_F = S_F[:,0].reshape(OF_shape)
            S2_F = S_F[:,1].reshape(OF_shape)
            S3_F = S_F[:,2].reshape(OF_shape)

            # Visualize Results of Successive + Gradient Combination
            fig, ((pl1, pl4), (pl2, pl5), (pl3, pl6)) = plt.subplots(3,2)
            fig.suptitle("pyiva vs. FastICA for Successive + Gradient Combination of X-Motion (Magnitude of Recovered Signal)")
            pl1.set_title("pyiva Recovered Source 1")
            pl1.imshow(cv2.cvtColor(np.float32(np.abs(S1)/np.abs(S1).max()),cv2.COLOR_GRAY2RGB))
            pl2.set_title("pyiva Recovered Source 2")
            pl2.imshow(cv2.cvtColor(np.float32(np.abs(S2)/np.abs(S2).max()),cv2.COLOR_GRAY2RGB))
            pl3.set_title("pyiva Recovered Source 3")
            pl3.imshow(cv2.cvtColor(np.float32(np.abs(S3)/np.abs(S3).max()),cv2.COLOR_GRAY2RGB))
            pl4.set_title("FastICA Recovered Source 1")
            pl4.imshow(cv2.cvtColor(np.float32(np.abs(S1_F)/np.abs(S1_F).max()),cv2.COLOR_GRAY2RGB))
            pl5.set_title("FastICA Recovered Source 2")
            pl5.imshow(cv2.cvtColor(np.float32(np.abs(S2_F)/np.abs(S2_F).max()),cv2.COLOR_GRAY2RGB))
            pl6.set_title("FastICA Recovered Source 3")
            pl6.imshow(cv2.cvtColor(np.float32(np.abs(S3_F)/np.abs(S3_F).max()),cv2.COLOR_GRAY2RGB))
            ##
            """

            ## DOUBLY SUCCESSIVE + GRADx2
            # Idea: Using Successive OF + GF fields, we can add a third source that may enable us to recover:
            # 1. Egomotion Field (Same in successive OFs + approx 0 in GF)
            # 2. Motion from OF 1
            # 3. Motion from OF 2
            # 4. Motion from OF 3
            # Results: Not much different from Successive + Grad
            ##

            ## LAPLACIAN
            # Idea: Employ the Laplacian Operator on the OF to try to eliminate ego-motion
            # See: demo_laplacian.py
            
            continue # Disabled so that the images aren't constantly overwritten

            # Old code for visualizing OF/GF, mostly for reference (pictures are in the repo)

            OF_magn = np.sqrt(np.sum(np.power(optical_flow1,2),axis=2))
            OF_magn = OF_magn / OF_magn.max()
            OF_dir = (np.arctan2(grad_flow[:,:,0],grad_flow[:,:,1]) + np.pi) / (2 * np.pi)

            GF_dir = (np.arctan2(grad_flow[:,:,0],grad_flow[:,:,1]) + np.pi) / (2*np.pi)
            
            dir_map = plt.get_cmap('hsv')
            GF_magn_GS = cv2.cvtColor(GF_magn,cv2.COLOR_GRAY2RGB)
            GF_dir_RGB = np.multiply(dir_map(GF_dir)[:,:,:3],255.0)
            GF_overlap_RGB = np.multiply(np.divide(GF_magn_GS,255.0),GF_dir_RGB)

            OF_combo = 25.5 * (np.floor((10*OF_magn)) + OF_dir)
            plt.imshow(cv2.cvtColor(OF_combo,cv2.COLOR_GRAY2RGB).astype('int'))
            plt.savefig(args.scene + "_of_combo_optical_flow.png")

            # Visualize results of basic ICA on combined mag/dir (found that this does not work very well)
            OF_ica = ica.fit_transform(OF_combo)
            OF_ica_inv = ica.inverse_transform(OF_ica)
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
