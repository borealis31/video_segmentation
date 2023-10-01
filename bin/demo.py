
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

            flo1 = flow_viz.flow_to_image(optical_flow1)
            flo2 = flow_viz.flow_to_image(optical_flow2)
            
            vis_img2 = image2
            vis_img2 = vis_img2[0].permute(1,2,0).cpu().numpy()
            
            img_flo = np.concatenate([vis_img2, flo1, flo2], axis=0)
            plt.imshow(img_flo / 255.0)
            plt.savefig("demo_flows.png")

            img_flo = np.concatenate([vis_img2, grad_flow_vis], axis=0)
            plt.imshow(img_flo / 255.0)
            plt.savefig("demo_grad.png")

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
            img = img[0].permute(1,2,0).cpu().numpy()"""

            """outliers = set(range(0, len(coords))) - set(inliers)
            for outlier in outliers:
                outlier = round(outlier)
                i1 = round(coords[outlier][1])*8
                i2 = round(coords[outlier][0])*8
                img[i1:(i1+8), i2:(i2+8) ] =  np.array([255, 0, 0])"""

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
            plt.savefig("foe_heatmap_blur.png")

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
