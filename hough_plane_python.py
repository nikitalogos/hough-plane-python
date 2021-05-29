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
    dot = np.abs(dot)
    return dot

def _fi_theta_depth_to_point(fi, theta, depth):
    '''Reconstruct point back from parameter space to 3D Euclidian space'''

    normal = np.array([
        math.sin(theta) * math.cos(fi),
        math.sin(theta) * math.sin(fi),
        math.cos(theta)
    ])
    return normal * depth

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def hough_planes(points, threshold,
                 fi_step=1, fi_bounds=[0,360], theta_step=1, theta_bounds=[0,180],
                 depth_grads=100, depth_start=0, verbose=True):
    '''

    :param points:
    :param threshold:
    :param fi_step:
    :param fi_bounds:
    :param theta_step:
    :param theta_bounds:
    :param depth_grads:
    :param depth_start:
    :param verbose:
    :return:
    '''

    fis = np.arange(fi_bounds[0], fi_bounds[1], fi_step)
    thetas = np.arange(theta_bounds[0], theta_bounds[1], theta_step)

    fis_len = len(fis)
    thetas_len = len(thetas)
    accum = np.zeros([fis_len, thetas_len, depth_grads], dtype=np.int)
    normals = _calc_normals(fis, thetas)

    points_max = np.max(points) * 2
    points_scaled = points * depth_grads / points_max

    fi_idxes = np.zeros([fis_len, thetas_len], dtype=np.int)
    for i in range(len(fis)):
        fi_idxes[i] = i
    fi_idxes = fi_idxes.flatten()
    theta_idxes = np.zeros([fis_len, thetas_len], dtype=np.int)
    for i in range(len(thetas)):
        theta_idxes[:, i] = i
    theta_idxes = theta_idxes.flatten()

    iterator = range(0, len(points))
    if CAN_USE_TQDM and verbose:
        iterator = tqdm(iterator)
    for k in iterator:
        point = points_scaled[k]

        dists = _dot_prod(point, normals).astype(np.int)
        dists = dists.flatten()

        # for i in range(len(fi_idxes)):
        #     accum[fi_idxes[i], theta_idxes[i], dists[i]] += 1
        accum[fi_idxes, theta_idxes, dists] += 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    points_best = []
    for i in range(len(fis)):
        for j in range(len(thetas)):
            for k in range(depth_start, depth_grads):
                v = accum[i, j, k]
                if v >= threshold:
                    points_best.append([i, j, k, v])
    points_best = np.array(points_best)
    if len(points_best) == 0:
        logger.warning('Failed to find hough planes: all points below threshold')
        return None, None


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_best[:,:3])
    cluster_idxes = pcd.cluster_dbscan(eps=3, min_points=5)

    clusters = {}
    for i in range(len(cluster_idxes)):
        idx = cluster_idxes[i]

        if not idx in clusters:
            clusters[idx] = []

        clusters[idx].append(points_best[i])

    logger.debug('Detect clusters in parameter space')
    planes_out = []
    for k, v in clusters.items():
        logger.debug(f'~~~~~~~~~~~~{k}~~~~~~~~~~~~')
        cluster = np.array(v, dtype=np.int)

        coords = cluster[:, :3]
        weights = cluster[:, 3]
        for i in range(3):
            coords[:, i] *= weights
        weight = len(weights)

        coord = np.sum(coords, axis=0) / np.sum(weights)
        logger.debug('coord', coord, 'weight', weight)

        fi = (fi_bounds[0] + coord[0]*fi_step) / 180 * np.pi
        theta = (theta_bounds[0] + coord[1]*theta_step) / 180 * np.pi
        depth = coord[2] / depth_grads * points_max
        point = _fi_theta_depth_to_point(fi, theta,depth)
        logger.debug('fi,theta,depth', fi,theta,depth)
        logger.debug('point', point)

        plane = np.concatenate([point, [weight]])
        planes_out.append(plane)
    planes_out = np.array(planes_out)
    logger.debug(f'~~~~~~~~~~~~~~~~~~~~~~~~')

    return planes_out, points_best