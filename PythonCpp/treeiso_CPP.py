"""Tree isolation from terrestrial laser scanning point clouds."""

import numpy as np
import numpy_indexed as npi
from scipy.spatial import cKDTree, ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from skimage import draw
import pandas as pd
import time

# Parameters
PR_REG_STRENGTH1 = 1.5 # lambda1  #1.5
PR_MIN_NN1 = 5 # K1: key parameter #5

PR_REG_STRENGTH2 = 30.  # 15. # lambda2: key parameter # 25.
PR_MIN_NN2 = 20  # K2: key parameter #20

PR_DECIMATE_RES1 = 0.05  # For speed optimization #0.01
PR_DECIMATE_RES2 = 0.1  # For speed optimization #0.05
PR_MAX_GAP = 1.5  # 2. # Max assumed point gap within a tree due to occlusion #1.5

PR_REL_HEIGHT_LENGTH_RATIO = 0.5  # rho #0.5
PR_VERTICAL_WEIGHT = 0.5  # w #0.5
PR_MIN_NN3 = 20  # trivial #20
PR_SCORE_CANDIDATE_THRESH = 0.7  # No need to change #0.7
PR_INIT_STEM_REL_LENGTH_THRESH = 1.5  # No need to change #1.5

PR_MAX_OUTLIER_GAP=3.2 # 3.0 #trivial: post-processing to remove isolated points with great gaps #3.2

silent = False
cut_pursuit_silent = False

def print_wFlag(text: str) -> None:
    if silent is not True:
        print(text)


# Try to import the C++ version first, fall back to Python if not available
USE_CPP = True
if USE_CPP:
    try:
        from cut_pursuit_py import perform_cut_pursuit
        USE_CPP = True
        print_wFlag("Using optimized C++ implementation of Cut-Pursuit")
    except ImportError as e:
        print(f"C++ implementation not available: {e}")
        try:
            from cut_pursuit_L0 import perform_cut_pursuit
            USE_CPP = False
            print_wFlag("Using Python implementation of Cut-Pursuit (slower)")
        except ImportError as e:
            print(f"Failed to import Cut-Pursuit: {e}")
            raise
else:
    try:
        from cut_pursuit_L0 import perform_cut_pursuit
        USE_CPP = False
        print_wFlag("Using Python implementation of Cut-Pursuit (slower)")
    except ImportError as e:
        print(f"Failed to import Cut-Pursuit: {e}")
        raise


def overlapping(conv_hull1, conv_hull2):
    """Calculate the 2D horizontal overlapping ratio between two convex hulls.

    Args:
        conv_hull1 (numpy.ndarray): First convex hull points, shape (N, 2)
        conv_hull2 (numpy.ndarray): Second convex hull points, shape (M, 2)

    Returns:
        float: Maximum overlapping ratio between the intersection and either hull
    """
    conv_hull1 = np.array(conv_hull1)
    conv_hull2 = np.array(conv_hull2)

    # Arbitrary scale factor to enlarge the mask image
    scale = 10

    # Combine hulls and normalize coordinates to start from (0,0)
    conv_hull_combo = np.vstack((conv_hull1, conv_hull2))
    conv_hull_combo_min = np.min(conv_hull_combo, axis=0)

    # Calculate digitization size
    digitize_size = np.ceil(
        np.max(conv_hull_combo[:, :2] - conv_hull_combo_min, axis=0) * scale
    ) + 10
    digitize_size = digitize_size.astype(int)
    digitize_size = [digitize_size[1], digitize_size[0]]  # Swap for height, width format

    # Scale and shift hull coordinates
    hull1_scaled = (conv_hull1 - conv_hull_combo_min) * scale
    hull2_scaled = (conv_hull2 - conv_hull_combo_min) * scale

    # Create binary masks using polygon
    conv_mask1 = np.zeros(digitize_size, dtype=np.uint8)
    conv_mask2 = np.zeros(digitize_size, dtype=np.uint8)

    # Extract x and y coordinates for both hulls
    r1, c1 = hull1_scaled[:, 1], hull1_scaled[:, 0]
    r2, c2 = hull2_scaled[:, 1], hull2_scaled[:, 0]

    # Create polygon masks using skimage
    rr1, cc1 = draw.polygon(r1, c1)
    rr2, cc2 = draw.polygon(r2, c2)

    conv_mask1[rr1, cc1] = 1
    conv_mask2[rr2, cc2] = 1

    # Calculate intersection mask
    conv_intersect_mask = conv_mask1 & conv_mask2

    # Calculate areas
    intersection_area = np.sum(conv_intersect_mask)
    mask1_area = np.sum(conv_mask1)
    mask2_area = np.sum(conv_mask2)
    if mask1_area<1e-9:
        mask1_area+=1e-9
    if mask2_area < 1e-9:
        mask2_area += 1e-9
    # Calculate maximum overlap ratio
    overlap_ratio = max(
        intersection_area / mask1_area,
        intersection_area / mask2_area
    )

    if np.isnan(overlap_ratio):
        overlap_ratio = 0
    return overlap_ratio


def init_segs(pcd):
    """Initialize segments from point cloud data."""
    pcd = pcd[:, :3] - np.mean(pcd[:, :3], axis=0)
    point_count = len(pcd)
    if point_count == 0:
        return False

    kdtree = cKDTree(pcd[:, :3])
    nn_D, nn_idx = kdtree.query(pcd, k=PR_MIN_NN1 + 1)

    # Remove self-connections
    indices = nn_idx[:, 1:]

    # Create edge list
    n_nodes = len(pcd)

    eu = np.repeat(np.arange(n_nodes), PR_MIN_NN1)
    ev = indices.ravel()

    # Edge weights following C++ implementation
    edge_weight = np.ones_like(eu, dtype=np.float32)

    in_component = perform_cut_pursuit(
        reg_strength=PR_REG_STRENGTH1,
        D=3,
        pc_vec=pcd.astype(np.float32),
        edge_weights=edge_weight,
        Eu=eu.astype(np.uint32),
        Ev=ev.astype(np.uint32),
        verbose=(not cut_pursuit_silent),
    )
    return in_component


def create_node_edges(points, k=10, max_distance=0.4):
    """Create node edges from point cloud data.
    
    Args:
        points: Point cloud data with [x, y, z, init_segs]
        k: Number of nearest neighbors
        max_distance: Max allowed distance for a component not to be considered as an outlier
    """
    _, centroids_idx, inverse_idx = np.unique(points[:, -1], return_index=True, return_inverse=True)
    _, v_group = npi.group_by(points[:, -1].astype(np.int32), np.arange(len(points[:, -1])))

    just_one = False
    centroids = np.array([np.mean(points[idx, :3], 0) for idx in v_group])
    # if centroids.shape[0]<2:
    #     return 0, 0, 0, 0, True

    kdtree = cKDTree(centroids[:, :3])

    _, indices = kdtree.query(centroids[:, :3], k=min(k + 1, len(centroids)))


    distance_matrix = np.zeros([len(centroids), len(centroids)]) - 1  #
    for i, v in enumerate(v_group):
        nn_idx = indices[i, 1:]
        tree = cKDTree(points[v, :3])
        distance_matrix[i, i] = 0
        for j, nv in enumerate(nn_idx):
            if distance_matrix[i, j] > 0:
                continue
            nn_dist = tree.query(points[v_group[nv], :3], k=1)
            distance_matrix[i, nv] = np.min(nn_dist)

    kdtree = cKDTree(points[:, :3])
    nn_D, nn_idx = kdtree.query(points[:, :3], k=min(k + 1, len(points)))
    indices = nn_idx[:, 1:]

    eu = np.repeat(np.arange(len(points)), k)
    ev = indices.ravel()

    eu_node = inverse_idx[eu]
    ev_node = inverse_idx[ev]

    distance_pairs = np.transpose([eu, ev, distance_matrix[eu_node, ev_node]])

    distance_pairs = distance_pairs[distance_pairs[:, -1] < max_distance]
    distance_pairs = distance_pairs[distance_pairs[:, -1] > -1]

    return centroids, distance_pairs, centroids_idx, inverse_idx


def intermediate_segs(pcd):
    """Create intermediate segments from point cloud data."""
    pcd[:, :3] = pcd[:, :3] - np.mean(pcd[:, :3], axis=0)
    centroids, distance_pairs, centroids_idx, centroids_inverse_idx = create_node_edges(pcd[:, :4], k=PR_MIN_NN2,
                                                                                        max_distance=PR_MAX_GAP)

    point_count = len(pcd)
    if point_count == 0:
        return False

    eu = distance_pairs[:, 0].astype(np.uint32)
    ev = distance_pairs[:, 1].astype(np.uint32)

    # Edge weights based on inverse distance
    if USE_CPP:
        edge_weight = 10 / ((distance_pairs[:, 2] + 0.01) / 0.01) * (PR_REG_STRENGTH2)
    else:
        edge_weight = 10 / ((distance_pairs[:, 2] + 0.01) / 0.01)

    in_component = perform_cut_pursuit(
        reg_strength=PR_REG_STRENGTH2,
        D=2,
        pc_vec=pcd[:, :2].astype(np.float32),
        edge_weights=edge_weight.astype(np.float32),
        Eu=eu,
        Ev=ev,
        verbose=(not cut_pursuit_silent),
    )
    return in_component


def trimmean(x, percent):
    """Calculate the robust mean by trimming outliers.
       Args:
        x: Input array
        percent: Percentage of data to trim from each end
    """
    x = np.asarray(x)
    if x.ndim > 2:
        raise ValueError("Input must be 1-D or 2-D array")
    if not 0 <= percent <= 100:
        raise ValueError("Percent must be between 0 and 100")
    if x.ndim == 1:
        n = len(x)
        k = int(round(n * percent / 100 / 2))
        return np.mean(np.sort(x)[k:n - k])
    else:  # 2-D array
        return np.array([trimmean(col, percent) for col in x.T])


def final_segs(pcd):
    """Create final segments from point cloud data.

    Args:
        pcd: Point cloud data with [x, y, z, init_segs, intermediate_segs]
    """
    pcd = np.concatenate([pcd, pcd[:, -1][:, np.newaxis]], axis=-1)

    # Group smallest clusters based on init_segs
    clusterIdx = pcd[:, -3]
    _, clusterUIdx, clusterUIdxInverse = np.unique(clusterIdx, return_index=True, return_inverse=True)
    clusterU, clusterVGroup = npi.group_by(clusterIdx.astype(np.int32), np.arange(len(clusterIdx)))
    centroids = np.vstack([np.mean(pcd[cluster_group_idx, :3], 0) for i, cluster_group_idx in enumerate(clusterVGroup)])

    clusterGroupIds = pcd[:, -1]  # Records final tree IDs per point

    # Group larger segments based on intermediate_segs
    clusterGroupMap = np.concatenate([centroids, pcd[clusterUIdx, -3:-1]], axis=-1)
    clusterMapU, clusterMapVGroup = npi.group_by(clusterGroupMap[:, -1].astype(np.int32),
                                                 np.arange(len(clusterGroupMap[:, -1])))

    clusterGroupIds = pcd[clusterUIdx, -1][clusterUIdxInverse]

    iter = 1
    toMergeIds = np.array([0, 0])
    prevToMergeIds = np.array([0])
    mergedRemainIds = []
    groupU = []
    groupVGroup = []
    groupFeatures = []

    # Iterative merging
    # Future: Merging one cluster each time can be inefficient and order-specific. A better way is to use graph with edge weights based on the weighted computed metrics, and optimize with the shortest path
    while (len(toMergeIds) != len(prevToMergeIds) and len(toMergeIds) > 0):
        prevToMergeIds = toMergeIds.copy()

        # Group tree segments based on temporal tree IDs (clusterGroupIds)
        groupU, groupV = np.unique(clusterGroupIds.astype(np.int32), return_inverse=True)
        _, groupVGroup = npi.group_by(clusterGroupIds.astype(np.int32), np.arange(len(clusterGroupIds)))

        # Extract features for each tree segment (e.g. convexhull)
        nGroups = len(groupVGroup)
        if nGroups == 1:
            return np.zeros(len(pcd))

        groupFeatures = np.zeros([nGroups, 5])
        groupHulls = [None] * nGroups
        for i in range(nGroups):
            groupPts = pcd[groupVGroup[i], :]
            groupFeatures[i, :3] = trimmean(groupPts[:, :3], 0.2)  # centroid
            groupFeatures[i, 3] = np.min(groupPts[:, 2])  # elevation
            groupFeatures[i, 4] = np.max(groupPts[:, 2]) - min(groupPts[:, 2])  # length
            try:
                hull = ConvexHull(groupPts[:, :2])
                groupHulls[i] = groupPts[hull.vertices, :2]
            except Exception:
                # If ConvexHull fails, create a simple bounding box instead
                mins = np.min(groupPts[:, :2], axis=0)
                maxs = np.max(groupPts[:, :2], axis=0)
                groupHulls[i] = np.array([
                    [mins[0], mins[1]],
                    [maxs[0], mins[1]],
                    [maxs[0], maxs[1]],
                    [mins[0], maxs[1]]
                ])
            # if len(groupPts) > 3:
            #     groupHulls[i] = groupPts[ConvexHull(groupPts[:,:2]).vertices,:2]

        # Search nearest k tree segments for each tree segments
        kdtree = cKDTree(groupFeatures[:, :2])
        groupNNCDs, groupNNIdxC = kdtree.query(groupFeatures[:, :2], k=min(len(groupFeatures), PR_MIN_NN3))
        groupNNCDs = np.transpose(groupNNCDs)[:, np.newaxis] if groupNNCDs.ndim == 1 else groupNNCDs

        sigmaD = np.mean(groupNNCDs[:, 1])  # Mean centroid distance between segments

        toMergeIds = np.zeros(nGroups, dtype=np.int32)

        # Find initial stem segments
        for i in range(nGroups):
            currentGroupFt = groupFeatures[i, 3:5]
            nnGroupId = groupNNIdxC[i, :]
            nnGroupFt = groupFeatures[nnGroupId, 3:5]
            if currentGroupFt[1]<1e-9:
                currentGroupFt[1]+=1e-9
            currentGroupRelHt = (currentGroupFt[0] - np.min(nnGroupFt[:, 0])) / currentGroupFt[1]
            # Set flags, for the first iteration, just merge longer segments first; merge remaining segments from the second iteration on
            if np.abs(currentGroupRelHt) > PR_REL_HEIGHT_LENGTH_RATIO:
                if (currentGroupFt[1] / np.median(nnGroupFt[:, 1])) > PR_INIT_STEM_REL_LENGTH_THRESH:
                    toMergeIds[i] = i
                else:
                    toMergeIds[i] = -i
        toMergeLogi = toMergeIds != 0
        remainIds = np.where(~toMergeLogi)[0]
        toMergeIds = toMergeIds[toMergeLogi]
        if (iter == 1) and np.sum(toMergeIds >= 0) > 0:
            toMergeIds = toMergeIds[toMergeIds > 0]
        else:
            toMergeIds = np.abs(toMergeIds)

        # Search nearest k tree segments for remaining segments
        kdtree = cKDTree(groupFeatures[remainIds, :2])
        _, groupNNIdx = kdtree.query(groupFeatures[toMergeIds, :2], k=min(PR_MIN_NN3, len(remainIds)))
        groupNNIdx = np.transpose(groupNNIdx)[:, np.newaxis] if groupNNIdx.ndim == 1 else groupNNIdx
        nNNs = groupNNIdx.shape[1]

        # Calculate similarity score based on gaps and overlaps
        for i, toMergeId in enumerate(toMergeIds):
            currentClusterCentroids = clusterGroupMap[np.concatenate([clusterMapVGroup[toMergeId]]), :]
            filterMetrics = np.zeros([nNNs, 5])
            for j in range(nNNs):
                remainId = remainIds[groupNNIdx[i, j]]
                line1Ends = [groupFeatures[toMergeId, 3], groupFeatures[toMergeId, 3] + groupFeatures[toMergeId, 4]]
                line2Ends = [groupFeatures[remainId, 3], groupFeatures[remainId, 3] + groupFeatures[remainId, 4]]
                lineSegs = [(line2Ends[1] - line1Ends[0]), (line1Ends[1] - line2Ends[0])]
                verticalOverlapRatio = np.min(lineSegs) / np.max(lineSegs)
                if groupHulls[toMergeId] is not None and groupHulls[remainId] is not None:
                    horizontalOverlapRatio = overlapping(groupHulls[toMergeId], groupHulls[remainId])
                else:
                    horizontalOverlapRatio = 0.0

                nnClusterCentroids = clusterGroupMap[np.concatenate([clusterMapVGroup[remainId]]), :]

                kdtree = cKDTree(nnClusterCentroids[:, :3])
                nnDs, idx = kdtree.query(currentClusterCentroids[:, :3], k=1)
                min3DSpacing = np.min(nnDs)
                min2DSpacing = np.linalg.norm(
                    np.mean(nnClusterCentroids[:, :2], 0) - np.mean(currentClusterCentroids[:, : 2], 0))
                filterMetrics[j, :] = np.array(
                    [horizontalOverlapRatio, verticalOverlapRatio, min3DSpacing, min2DSpacing, remainId])
            filterMetrics[filterMetrics[:, 1] <= 0, 1] = 0
            score = np.exp(
                -np.power(1 - filterMetrics[:, 0], 2) - PR_VERTICAL_WEIGHT * np.power(1 - filterMetrics[:, 1],
                                                                                      2) - np.power(
                    np.min(filterMetrics[:, 2:4], 1) / sigmaD, 2))

            scoreSortI = score.argsort()[::-1]
            scoreSort = score[scoreSortI]
            if scoreSort[0] == 0:
                continue
            scoreSortRatio = scoreSort / scoreSort[0]  # Normalize score based on the largest score
            scoreSortCandidateIdx = np.where(scoreSortRatio > PR_SCORE_CANDIDATE_THRESH)[
                0]  # Filter only the large scores as candidates
            nScoreSortCandidateIdx = len(scoreSortCandidateIdx)
            if nScoreSortCandidateIdx == 1:
                mergeNNId = groupU[
                    int(filterMetrics[scoreSortI[scoreSortCandidateIdx[0]], -1])]  # Tree segment ID to merge
            elif nScoreSortCandidateIdx > 1:
                filterMinSpacingIdx = np.argmin(filterMetrics[scoreSortI[
                    scoreSortCandidateIdx], 2])  # If there are more than one candidates, choose the one with minimal 3D point gap to the current tree segment
                mergeNNId = groupU[int(filterMetrics[scoreSortI[filterMinSpacingIdx], -1])]  # Tree segment ID to merge
            else:
                continue
            clusterGroupIds[groupVGroup[toMergeIds[i]]] = mergeNNId  # Update the point-level tree Ids

        mergedRemainIds.extend(groupU[remainIds])
        # Re-group tree segments based on updated tree IDs
        clusterGroupMap[:, -1] = clusterGroupIds[clusterUIdx]
        clusterMapU, clusterMapVGroup = npi.group_by(clusterGroupMap[:, -1].astype(np.int32),
                                                     np.arange(len(clusterGroupMap[:, -1])))
        iter = iter + 1

    # For the remaining unmerged non-stem segments, merge each to the nearest segment with minimal 3D point gap
    unmergeIds = np.setdiff1d(groupU, mergedRemainIds)
    if len(unmergeIds) > 0:
        mergedRemainIds = np.unique(mergedRemainIds)

        unmergeIds = np.where(np.isin(clusterMapU, unmergeIds))[0]
        mergedRemainIds = np.where(np.isin(clusterMapU, mergedRemainIds))[0]

        kdtree = cKDTree(groupFeatures[mergedRemainIds, :2])
        _, groupNNIdx = kdtree.query(groupFeatures[unmergeIds, :2], k=min(PR_MIN_NN3, len(mergedRemainIds)))
        groupNNIdx = np.transpose(groupNNIdx)[:, np.newaxis] if groupNNIdx.ndim == 1 else groupNNIdx
        nNNs = groupNNIdx.shape[1]

        for i, unmergeId in enumerate(unmergeIds):
            currentClusterCentroids = clusterGroupMap[np.concatenate([clusterMapVGroup[unmergeId]]), :]
            filterMetrics = np.zeros([nNNs, 2])
            for j in range(nNNs):
                mergedRemainId = mergedRemainIds[groupNNIdx[i, j]]
                nnClusterCentroids = clusterGroupMap[np.concatenate([clusterMapVGroup[mergedRemainId]]), :]

                kdtree = cKDTree(currentClusterCentroids[:, :3])
                nnDs, idx = kdtree.query(nnClusterCentroids[:, :3])

                min3DSpacing = min(nnDs)
                filterMetrics[j, :] = np.array([min3DSpacing, mergedRemainId])

            filterMinSpacingIdx = np.argmin(filterMetrics[:, 0])
            mergeNNId = groupU[int(filterMetrics[filterMinSpacingIdx, -1])]
            clusterGroupIds[groupVGroup[unmergeIds[i]]] = mergeNNId

    _, clusterGroupIds = np.unique(clusterGroupIds, return_inverse=True)
    return clusterGroupIds


def decimate_pcd(columns, min_res):
    """Resample the point cloud to a lower resolution.

    Args:
        columns: Point cloud data columns
        min_res: Desired minimum resolution
    """
    _, block_idx_uidx, block_inverse_idx = np.unique(np.floor(columns[:, :3] / min_res).astype(np.int32), axis=0,
                                                     return_index=True, return_inverse=True)
    return block_idx_uidx, block_inverse_idx

def isolate_gaps(pcd,max_gap,search_K=20):
    pcd = pcd[:, :3] - np.mean(pcd[:, :3], axis=0)
    point_count = len(pcd)
    if point_count == 0:
        return False

    K=min(search_K,len(pcd))
    kdtree = cKDTree(pcd[:, :3])
    nn_D, nn_idx = kdtree.query(pcd[:, :3], k=K)

    # Remove self-connections
    indices = nn_idx[:, 1:]
    nn_D = nn_D[:, 1:]

    # Create edge list
    eu = np.repeat(np.arange(len(pcd)), K-1)
    ev = indices.ravel()
    nn_D=nn_D.ravel()
    inlier_ind=nn_D<max_gap
    eu=eu[inlier_ind]
    ev=ev[inlier_ind]

    adjacency_matrix = csr_matrix((np.ones(len(eu), dtype=int), (eu, ev)), shape=(point_count, point_count))
    n_components, labels = connected_components(csgraph=adjacency_matrix,directed=False,connection='weak',return_labels=True)

    return labels

def tree_isolator(pcd):
    print_wFlag('\n\n--- TREE SEGMENTATION ---')

    pcd = pcd - np.mean(pcd, axis=0)

    start_glob = time.perf_counter()

    # Initial segmentation in 3D
    print_wFlag('-Initial segmentation in 3D-')
    start = time.perf_counter()
    dec_idx_uidx, dec_inverse_idx = decimate_pcd(pcd, PR_DECIMATE_RES1)
    pcd_dec = pcd[dec_idx_uidx]
    init_labels = init_segs(pcd_dec)
    # print('np.unique(init_labels)', np.unique(init_labels[dec_inverse_idx]))
    stop = time.perf_counter()
    print_wFlag(f'Operation took: {stop - start:.2f} s')

    if np.unique(init_labels).shape[0] < 50:
        return np.full(pcd_dec.shape[0], -1)

    # Intermediate segmentation in 2D
    print_wFlag('-Intermediate segmentation in 2D-')
    start = time.perf_counter()
    dec_idx_uidx2, dec_inverse_idx2 = decimate_pcd(pcd, PR_DECIMATE_RES2)
    pcd_dec2 = pcd[dec_idx_uidx2]


    intermediate_labels = intermediate_segs(
        np.concatenate([pcd_dec2, init_labels[dec_inverse_idx][dec_idx_uidx2][:, np.newaxis]], axis=-1)
    )
    stop = time.perf_counter()
    print_wFlag(f'Operation took: {stop - start:.2f} s')

    if np.unique(intermediate_labels).shape[0] < 3:
        return intermediate_labels[dec_inverse_idx2]


    # Final segmentation based on segment similarity
    print_wFlag('-Final segmentation based on segment similarity-')
    start = time.perf_counter()

    final_labels = final_segs(
        np.concatenate([
            pcd_dec,
            init_labels[:, np.newaxis],
            intermediate_labels[dec_inverse_idx2][dec_idx_uidx][:, np.newaxis]
        ], axis=-1)
    )

    stop = time.perf_counter()
    print_wFlag(f'Operation took: {stop - start:.2f} s')
    stop_glob = time.perf_counter()

    print_wFlag(f'Whole  segmentation operation took: {stop_glob - start_glob:.2f} s')


    #some more processing here


    if_isolate_outlier = True

    if if_isolate_outlier:
        connected_labels=isolate_gaps(pcd_dec, PR_MAX_OUTLIER_GAP)

        _,final_labels=np.unique(np.transpose([final_labels,connected_labels]),axis=0,return_inverse=True)

        # connected_labels = connected_labels[dec_inverse_idx]
        final_labels = final_labels[dec_inverse_idx]

        return final_labels
    else:
        final_labels = final_labels[dec_inverse_idx]
        return final_labels