import numpy as np
import math
import random 

RANSAC_MINIMUM_DATAPOINTS = 2
RANSAC_NUMBER_ITERATIONS = 50
RANSAC_MINIMUM_ERROR_DISTANCE = 100
RANSAC_MINIMUM_RATIO_INLIERS = 0.60
RANSAC_MINIMUM_ERROR_ANGLE = 15
RANSAC_RATIO_INCREASE_ETA = 0.0001

def foe(c0, c1):
    c0 = c0[0].permute(1, 2, 0).cpu().numpy()
    c1 = c1[0].permute(1, 2, 0).cpu().numpy()

    c0 = c0.reshape((c0.shape[0]*c0.shape[1], 2))
    c1 = c1.reshape((c1.shape[0]*c1.shape[1], 2))
    coords = np.hstack((c0, c1))
    return coords

def l2_norm_optimization(a_i, b_i, c_i, w_i=None):
    """Solve l2-norm optimization problem."""

    # Non-Weighted optimization:
    if w_i is None:
        aux1 = -2 * ((np.sum(np.multiply(b_i, b_i))) * (
            np.sum(np.multiply(a_i, a_i))) / float(
            np.sum(np.multiply(a_i, b_i))) - (np.sum(np.multiply(a_i, b_i))))
        aux2 = 2 * ((np.sum(np.multiply(b_i, b_i))) * (
            np.sum(np.multiply(a_i, c_i))) / float(
            np.sum(np.multiply(a_i, b_i))) - (np.sum(np.multiply(b_i, c_i))))

        x0 = aux2 / float(aux1)

        y0 = (-(np.sum(np.multiply(a_i, c_i))) - (
            np.sum(np.multiply(a_i, a_i))) * x0) / float(
            np.sum(np.multiply(a_i, b_i)))

    # Weighted optimization:
    else:
        aux1 = -2 * ((np.sum(np.multiply(np.multiply(b_i, b_i), w_i))) * (
            np.sum(np.multiply(np.multiply(a_i, a_i), w_i))) / float(
            np.sum(np.multiply(np.multiply(a_i, b_i), w_i))) - (
                         np.sum(np.multiply(np.multiply(a_i, b_i), w_i))))
        aux2 = 2 * ((np.sum(np.multiply(np.multiply(b_i, b_i), w_i))) * (
            np.sum(np.multiply(np.multiply(a_i, c_i), w_i))) / float(
            np.sum(np.multiply(np.multiply(a_i, b_i), w_i))) - (
                        np.sum(np.multiply(np.multiply(b_i, c_i), w_i))))

        x0 = aux2 / float(aux1)

        y0 = (-(np.sum(np.multiply(np.multiply(a_i, c_i), w_i))) - (
            np.sum(np.multiply(np.multiply(a_i, a_i), w_i))) * x0) / float(
            np.sum(np.multiply(np.multiply(a_i, b_i), w_i)))

    # return resulting point
    return (x0, y0)

def select_subset(OFVectors, seed=0):
    """Select a subset of a given set."""
    random.seed(seed)
    subset = np.array([]).reshape(0, 4)
    for i in range(RANSAC_MINIMUM_DATAPOINTS):
        idx = random.randint(0, (OFVectors.shape)[0] - 1)
        subset = np.vstack((subset, np.array([OFVectors[idx]])))
    return subset

def fit_model(subset):
    """Return a solution for a given subset of points."""
    # Initialize some empty variables
    a_i = np.array([])
    b_i = np.array([])
    c_i = np.array([])

    # Save the lines coeficients of the form a*x + b*y + c = 0 to the variables
    for i in range(subset.shape[0]):
        a1, b1, c1, d1 = subset[i]

        pt1 = (a1, b1)
        # So we don't divide by zero
        if (a1 - c1) == 0:
            continue
        a = float(b1 - d1) / float(a1 - c1)
        b = -1
        c = (b1) - a * a1

        denominator = float(a ** 2 + 1)

        a_i = np.append(a_i, a / denominator)
        b_i = np.append(b_i, b / denominator)
        c_i = np.append(c_i, c / denominator)

    # Solve a optimization problem with Minimum Square distance as a metric
    (x0, y0) = l2_norm_optimization(a_i, b_i, c_i)

    # Return FOE
    return (x0, y0)

def find_angle_between_lines(x0, y0, a1, b1, c1, d1):
    """Finds the angle between two lines."""

    # Line 1 : line that passes through (x0,y0) and (a1,b1)
    # Line 2 : line that passes through (c1,d1) and (a1,b1)

    angle1 = 0
    angle2 = 0
    if (a1 - x0) != 0:
        angle1 = float(b1 - y0) / float(a1 - x0)
    if (a1 - c1) != 0:
        angle2 = float(b1 - d1) / float(a1 - c1)

    # Get angle in degrees
    angle1 = math.degrees(math.atan(angle1))
    angle2 = math.degrees(math.atan(angle2))

    ang_diff = angle1 - angle2
    # Find angle in the interval [0,180]
    if math.fabs(ang_diff) > 180:
        ang_diff = ang_diff - 180

    # Return angle between the two lines
    return ang_diff

def find_inliers_outliers(x0, y0, OFVectors):
    """Find set of inliers and outliers of a given set of optical flow vectors and the estimated FOE."""
    # Initialize some varaiables
    inliers = np.array([])
    nbr_inlier = 0

    # Find inliers with the angle method

    # For each vector
    for i in range((OFVectors.shape)[0]):
        a1, b1, c1, d1 = OFVectors[i]
        # Find the angle between the line that passes through (x0,y0) and (a1,b1) and the line that passes through (c1,d1) and (a1,b1)
        ang_diff = find_angle_between_lines(x0, y0, a1, b1, c1, d1)
        # If the angle is below a certain treshold consider it a inlier
        if -RANSAC_MINIMUM_ERROR_ANGLE < ang_diff < RANSAC_MINIMUM_ERROR_ANGLE:
            # Increment number of inliers and add save it
            nbr_inlier += 1
            inliers = np.append(inliers, i)
    # Compute the ratio of inliers to overall number of optical flow vectors
    ratioInliersOutliers = float(nbr_inlier) / (OFVectors.shape)[0]

    # Return set of inliers and ratio of inliers to overall set
    return inliers, ratioInliersOutliers

def RANSAC(OFVectors):
    """Estimate the FOE of a set of optical flow (OF) vectors using a form of RANSAC method."""
    # Initialize some variables
    FOE = (0, 0)
    savedRatio = 0
    inliersModel = np.array([])

    # Repeat iterations for a number of times
    for i in range(RANSAC_NUMBER_ITERATIONS):
        # Randomly select initial OF vectors
        subset = select_subset(OFVectors)
        # Estimate a FOE for the set of OF vectors
        (x0, y0) = fit_model(subset)
        # Find the inliers of the set for the estimated FOE
        inliers, ratioInliersOutliers = find_inliers_outliers(x0, y0, OFVectors)
        # Initialize some varaibles
        iter = 0
        ratioInliersOutliers_old = 0
        # While the ratio of inliers keeps on increasing
        while ((inliers.shape)[
                   0] != 0 and ratioInliersOutliers - ratioInliersOutliers_old > RANSAC_RATIO_INCREASE_ETA):
            # Repeat iterations for a number of times
            if iter > RANSAC_NUMBER_ITERATIONS:
                break
            iter += 1
            # Select a new set of OF vectors that are inliers tot he estimated FOE
            for i in range((inliers.shape)[0]):
                subset = np.vstack(
                    (subset, np.array([OFVectors[int(inliers[i])]])))
            # Estimate a FOE for the new set of OF vectors
            (x0, y0) = fit_model(subset)
            # Save the previous iteration ratio if inliers
            ratioInliersOutliers_old = ratioInliersOutliers
            # Find the inliers of the set for the estimated FOE
            inliers, ratioInliersOutliers = find_inliers_outliers(x0, y0, OFVectors)

            # If ratio of inliers is bigger than the previous iterations, save current solution
            if savedRatio < ratioInliersOutliers:
                savedRatio = ratioInliersOutliers
                inliersModel = inliers
                FOE = (x0, y0)
            # If ratio is acceptable, stop iterating and return the found solution
            if savedRatio > RANSAC_MINIMUM_RATIO_INLIERS and RANSAC_MINIMUM_RATIO_INLIERS != 0:
                break
        # If ratio is acceptable, stop iterating and return the found solution
        if savedRatio > RANSAC_MINIMUM_RATIO_INLIERS and RANSAC_MINIMUM_RATIO_INLIERS != 0:
            break

    # Return the estimated FOE, the found inliers ratio and the set of inliers
    return FOE, savedRatio, inliersModel

    """Computes the weighted mean of points given the weights."""
    num_x = 0
    num_y = 0
    den = 0
    # for each point, calculate the x and y mean
    for i in range(points.shape[1]):
        num_x = num_x + weights[i] * points[0][i]
        num_y = num_y + weights[i] * points[1][i]
        den = den + weights[i]
    # return the weighted means of x and y
    return np.array([num_x/den, num_y/den])
