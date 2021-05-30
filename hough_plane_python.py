import numpy as np
import open3d as o3d
import math
import logging
logger = logging.getLogger('Hough Plane')

CAN_USE_TQDM = False
try:
    from tqdm import tqdm
    CAN_USE_TQDM = True
except:
    pass

def _calc_normals(fis_deg, thetas_deg):
    '''Compute 2d array of vectors with length 1,
    each representing direction at angles φ and θ in spherical coordinate system'''

    fis_len = len(fis_deg)
    thetas_len = len(thetas_deg)

    normals = np.zeros((fis_len, thetas_len, 3), dtype=np.float)

    fis_rad = fis_deg / 180 * np.pi
    thetas_rad = thetas_deg / 180 * np.pi

    for i in range(fis_len):
        fi = fis_rad[i]
        for j in range(thetas_len):
            theta = thetas_rad[j]
            normal = np.array([
                math.sin(theta) * math.cos(fi),
                math.sin(theta) * math.sin(fi),
                math.cos(theta)
            ])
            normals[i, j] = normal

    return normals

def _dot_prod(point,normals):
    '''For one point compute projections of this point to all vectors in array "normals"'''

    x,y,z = point
    xx,yy,zz = normals[:,:,0], normals[:,:,1], normals[:,:,2]
    dot = x*xx + y*yy + z*zz
    return dot

def _fi_theta_depth_to_point(fi, theta, depth):
    '''Reconstruct point back from parameter space to 3D Euclidian space'''

    normal = np.array([
        math.sin(theta) * math.cos(fi),
        math.sin(theta) * math.sin(fi),
        math.cos(theta)
    ])
    return normal * depth

def vectors_len(vectors):
    '''Get lengths for array of 3D vectors'''

    vectors_sqr = vectors**2
    vectors_sum = np.sum(vectors_sqr, axis=1)
    vectors_sqrt = vectors_sum**0.5
    return vectors_sqrt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def hough_planes(points, threshold, use_tqdm=True,
                 fi_step=1, fi_bounds=(0, 360),
                 theta_step=1, theta_bounds=(0, 180),
                 depth_steps=100, depth_bounds=(0, None), depth_start_step=3,
                 dbscan_eps=3, dbscan_min_points=5,
                 ):
    '''Detects planes in 3D point clouds using Hough Transform algorithm.

    Algorithm transforms 3D points (e. g. [1.0, -2.0, 0.0]) into parameter space
    with axes φ (fi), θ (theta), d.
    These 3 parameters represent planes in 3D space. Angles φ and θ define normal vector,
    and d defines distance from zero to plane, orthogonal to this vector.

    After filling accumulator, we clusterize it via dbscan algorithm.
    Then, for every cluster we find its center of mass to get plane representation
    and its size for comparsion with other clusters.

    ---
    General parameters:
    :param points: 3D points as numpy array with shape=(?, 3) and dtype=np.float
    :param threshold: Minimal value in accumulator cell.
    :param use_tqdm: This flag defines whether use tqdm or not for the slowest part of algorithm
    which is filling accumulator with values point by point.

    ---
    Parameters φ, θ and d make up a 3D tensor.
    You can specify bounds and accuracy along each axis:

    :param fi_step: Step in degrees along φ axis in parameter space
    :param fi_bounds: Pair of two values: lower and upper bound for φ axis in degrees

    :param theta_step: Step in degrees along θ axis in parameter space
    :param theta_bounds: Pair of two values: lower and upper bound for φ axis in degrees

    :param depth_steps: Number of values in accumulator along d axis
    :param depth_bounds: Pair of two values: lower and upper bound for d axis.
    By default lower bound is 0, and upper bound is computed from point cloud
    so that no one point would go out of accumulator bounds.
    However, this results in almost unused upper half of accumulator along axis d.

    :param depth_start_step: Hough space is not uniform: it's density proportional to 1/4π*d.
    That's why lower slices of accumulator along axis d usually contain a lot of mess.
    To get rid of it, set this parameter to some small value: 3..5 (default is 3).

    !!Attention!! This only needed if you set lower bound of depth_bounds to 0.
    Instead, set depth_start_step to 0 in order to not to lose meaningful data!

    ---
    Dbscan parameters. Parameters passed to open3d.geometry.PointCloud.cluster_dbscan():
    :param dbscan_eps: Minimal distance between points in cluster.
    :param dbscan_min_points: Minimal points in cluster.

    ---
    :return: Returns 2 objects:
    -   np.array shape=(?,4) of planes, each represented by 3D point (x,y,z) that belongs to plane,
        which is also a vector normal to that plane;
        and s - size of cluster in parameter space that was collapsed into that plane;
        resulting in [x,y,z,s] vector for each plane.
    -   points in accumulator which value (v) is above threshold.
        Points format is (?,4), where each point has format [φ, θ, d, v]
    '''

    assert(type(points) == np.ndarray)
    assert(len(points.shape) == 2 and points.shape[1] == 3)

    assert(threshold >= 0)

    assert(fi_step > 0)
    assert(theta_step > 0)

    assert(depth_steps > 0 and type(depth_steps) == int)
    assert(depth_bounds[0] >= 0)
    assert(depth_start_step >= 0 and type(depth_start_step) == int)


    fis = np.arange(fi_bounds[0], fi_bounds[1], fi_step)
    thetas = np.arange(theta_bounds[0], theta_bounds[1], theta_step)

    fis_len = len(fis)
    thetas_len = len(thetas)
    accum = np.zeros([fis_len, thetas_len, depth_steps], dtype=np.int)
    normals = _calc_normals(fis, thetas)

    depth_bounds = list(depth_bounds)
    if depth_bounds[0] > 0:
        mask = vectors_len(points) > depth_bounds[0]
        points = points[mask]

    if depth_bounds[1] is None:
        depth_bounds[1] = np.max(points) * 2
    logger.debug(f'depth_bounds: {depth_bounds}')

    depth_skipped_steps = depth_bounds[0] / (depth_bounds[1] - depth_bounds[0]) * depth_steps
    depth_total_steps = depth_steps + depth_skipped_steps
    points_scaled = points / depth_bounds[1] * depth_total_steps

    fi_idxes = np.zeros([fis_len, thetas_len], dtype=np.int)
    for i in range(len(fis)):
        fi_idxes[i] = i
    fi_idxes = fi_idxes.flatten()
    theta_idxes = np.zeros([fis_len, thetas_len], dtype=np.int)
    for i in range(len(thetas)):
        theta_idxes[:, i] = i
    theta_idxes = theta_idxes.flatten()

    iterator = range(0, len(points))
    if CAN_USE_TQDM and use_tqdm:
        iterator = tqdm(iterator)
    for k in iterator:
        point = points_scaled[k]

        dists = _dot_prod(point, normals) - depth_skipped_steps
        dists = dists.astype(np.int)
        dists = dists.flatten()

        mask = (dists >= 0) * (dists < depth_steps)

        fi_idxes_ = fi_idxes[mask]
        theta_idxes_ = theta_idxes[mask]
        dists = dists[mask]

        accum[fi_idxes_, theta_idxes_, dists] += 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    points_best = []
    for i in range(len(fis)):
        for j in range(len(thetas)):
            for k in range(depth_start_step, depth_steps):
                v = accum[i, j, k]
                if v >= threshold:
                    points_best.append([i, j, k, v])
    points_best = np.array(points_best)
    if len(points_best) == 0:
        logger.warning('Failed to find hough planes: all points below threshold')
        return None, None


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_best[:,:3])
    cluster_idxes = pcd.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points)

    clusters = {}
    for i in range(len(cluster_idxes)):
        idx = cluster_idxes[i]

        if not idx in clusters:
            clusters[idx] = []

        clusters[idx].append(points_best[i])

    if -1 in clusters:
        del clusters[-1]
    if len(clusters.keys()) == 0:
        logger.warning('Failed to clusterize points in parameter space!')
        return None, points_best

    logger.debug('Detect clusters in parameter space')
    planes_out = []
    for k, v in clusters.items():
        logger.debug(f'~~~~~~~~~~~~{k}~~~~~~~~~~~~')
        cluster = np.array(v, dtype=np.int)

        coords = cluster[:, :3]
        weights = cluster[:, 3]
        for i in range(3):
            coords[:, i] *= weights
        cluster_size = len(weights)

        coord = np.sum(coords, axis=0) / np.sum(weights)
        logger.debug(f'coord={coord}, cluster_size={cluster_size}')

        fi = (fi_bounds[0] + coord[0]*fi_step) / 180 * np.pi
        theta = (theta_bounds[0] + coord[1]*theta_step) / 180 * np.pi
        depth = (coord[2] + depth_skipped_steps) / depth_total_steps * depth_bounds[1]
        point = _fi_theta_depth_to_point(fi, theta,depth)
        logger.debug(f'fi,theta,depth = ({fi},{theta},{depth})')
        logger.debug(f'plane point: {point}')

        plane = np.concatenate([point, [cluster_size]])
        planes_out.append(plane)
    planes_out = np.array(planes_out)
    logger.debug(f'~~~~~~~~~~~~~~~~~~~~~~~~')

    return planes_out, points_best