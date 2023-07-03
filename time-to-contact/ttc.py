import numpy as np

def get_ttc(coords, optical_flows, foe):
    y = (coords[:,:,0] - foe[0])**2 + (coords[:,:,1] - foe[1])**2
    dy = optical_flows[:,:,0]**2 + optical_flows[:,:,0]**2
    ttc = np.sqrt(y/dy)
    return ttc
