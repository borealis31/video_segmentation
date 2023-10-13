
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
        images = os.listdir(args.path)
        for image in images:
            if not(image.endswith(".png")) and not(image.endswith(".jpg")):
                images.remove(image)
        images = sorted(images)
        images = [args.path + image for image in images]

        if len(images) < 3:
            print("Error: Expected at least 3 images")
            return

        # Load three images based on ordering in name (chronological)
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
            grad_flow_vis = flow_viz.flow_to_image(grad_flow)

            gf_magn = np.sqrt(np.sum(np.power(grad_flow,2),axis=2))
            gf_magn = np.divide(np.multiply(gf_magn, 255),np.max(gf_magn))
            gf_magn_rgb = cv2.cvtColor(gf_magn,cv2.COLOR_GRAY2RGB)

            dir_map = plt.get_cmap('hsv')
            gf_dir = np.divide(np.add(np.arctan2(grad_flow[:,:,0],grad_flow[:,:,1]),np.pi),2*np.pi)
            gf_dir = np.multiply(dir_map(gf_dir)[:,:,:3],255.0)
            
            gf_overlap = np.multiply(np.divide(gf_magn_rgb,255.0),gf_dir)
            ica = FastICA(n_components=1)
            ica.fit(gf_magn)
            gf_ica = ica.fit_transform(gf_magn)
            gf_ica_inv = ica.inverse_transform(gf_ica)
            gf_ica_inv = 255.0*gf_ica_inv/np.max(gf_ica_inv)
            gf_iso = gf_magn - gf_ica_inv

            plt.imshow(cv2.cvtColor(gf_iso,cv2.COLOR_GRAY2RGB).astype('int'))
            plt.savefig("inv_iso.png")


            print(gf_magn.shape)
            print(gf_ica.shape)

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
            fig.savefig("demo_grad_magn.png")

            pl1.cla()
            pl2.cla()

            pl3 = fig.add_axes([0.15, 0.1, 0.7, 0.025])
            pl1.imshow(vis_img2.astype('int'))
            pl2.imshow(gf_dir.astype('int'))
            fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=360), cmap=mpl.cm.hsv),
                         cax=pl3, orientation="horizontal",label="Gradient Direction [deg]")
            fig.suptitle("Optical Flow Direction")
            fig.savefig("demo_grad_dir.png")

            pl1.cla()
            pl2.cla()

            pl1.imshow(vis_img2.astype('int'))
            pl2.imshow(gf_overlap.astype('int'))
            fig.suptitle("Optical Flow Magnitude and Direction Overlayed")
            fig.savefig("demo_grad_overlap.png")
            
            continue

            """ Commented 1/10, Keeping for Future Reference
            FoE Sampling Fails to Provide Metric for Objects Moving Parallel to the Camera
            # get foe
            # NOTE: we can kernel sample coords0 and coords1 to get subset and find FoE for each subset
            # then have the results appended to a list of FoEs
            foe_size = int(args.sample)
            foe_step = 2
            foe_set = []
            #print(coords0.size())
            for i in range(0, coords0.size()[2] - foe_size - 1, foe_step):
                for j in range(0, coords0.size()[3] - foe_size - 1, foe_step):
                    # Sliding Window Implementation
                    foe_sample0 = coords0[0, :, i:i+foe_size, j:j+foe_size]
                    foe_sample1 = coords1[0, :, i:i+foe_size, j:j+foe_size]
                    foe, inlier_ratio, inliers = RANSAC(foe_sample0, foe_sample1)
                    foe = [(foe[0])*8, (foe[1])*8]
                    foe_set.append(foe)
                    print(len(foe_set))

            # visualization
            img = image1
            img = img[0].permute(1,2,0).cpu().numpy()

            flo = flow_viz.flow_to_image(optical_flow)

            img_flo = np.concatenate([img, flo], axis=0)
            #plt.plot(foe[0], foe[1], marker="v", color="red")
            plt.imshow(img_flo / 255.0)
            #plt.show()
            plt.savefig("demo.png")

            foe_x, foe_y = zip(*foe_set)
            plt.cla()
            heatmap, xe, ye = np.histogram2d(foe_x, foe_y, bins=(np.ceil([coords0.size()[3]*2, coords0.size()[2]*2])).astype(int),range=[[0, img.shape[1]],[0, img.shape[0]]])
            ex = [xe[0], xe[-1], ye[0], ye[-1]]
            if ex[0] > 0:
                ex[0] = 0
            if ex[1] < img.shape[1]:
                ex[1] = img.shape[1]
            if ex[2] > 0:
                ex[2] = 2
            if ex[3] < img.shape[0]:
                ex[3] = img.shape[0]
            heatmap = np.flip(heatmap.T,0)
            plt.imshow(heatmap, extent=ex, cmap=mpl.colormaps['turbo'])
            #plt.scatter(foe_x,foe_y)
            plt.imshow(img / 255.0, alpha=0.5)
            plt.savefig("foe_heatmap.png")

            plt.cla()
            blurred_hm = cv2.GaussianBlur(heatmap, (7,7), cv2.BORDER_DEFAULT)
            plt.imshow(blurred_hm, extent=ex, cmap=mpl.colormaps['turbo'])
            plt.imshow(img / 255.0, alpha=0.5)
            plt.savefig("foe_heatmap_blur.png")"""

            """outliers = set(range(0, len(coords))) - set(inliers)
            for outlier in outliers:
                outlier = round(outlier)
                i1 = round(coords[outlier][1])*8
                i2 = round(coords[outlier][0])*8
                img[i1:(i1+8), i2:(i2+8) ] =  np.array([255, 0, 0])"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="../optical-flow/models/raft-things.pth",help="restore checkpoint")
    parser.add_argument('--path', default="example_frames/",help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--sample', default="16",help="use a square of sample to analyze FoEs")
    args = parser.parse_args()

    main(args)
