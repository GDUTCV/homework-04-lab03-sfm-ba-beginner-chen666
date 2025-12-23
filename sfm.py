import os
import numpy as np
import cv2

from tqdm import tqdm
import json
import open3d as o3d
import random
from scipy.optimize import least_squares

from preprocess import get_selected_points2d, get_camera_intrinsics
from preprocess import SCENE_GRAPH_FILE, RANSAC_MATCH_DIR, RANSAC_ESSENTIAL_DIR, HAS_BUNDLE_ADJUSTMENT, RESULT_DIR
from bundle_adjustment import compute_ba_residuals


def get_init_image_ids(scene_graph: dict) -> (str, str):

    max_pair = [None, None]
    max_score = 0

    image_ids = sorted(list(scene_graph.keys()))

    for i in range(len(image_ids)):
        img1 = image_ids[i]
        neighbors = scene_graph[img1]
        for img2 in neighbors:
            if img1 >= img2: continue

            # 解析文件名中的数字索引
            try:
                idx1 = int(''.join(filter(str.isdigit, img1)))
                idx2 = int(''.join(filter(str.isdigit, img2)))
                gap = abs(idx1 - idx2)
            except:
                gap = 0


            # 间隔太小(<=2)会导致基线太短，模型压扁 -> 跳过
            # 间隔太大(>15)会导致匹配太少，模型甚至无法连接 -> 跳过
            if gap < 3 or gap > 15:
                continue

            matches = load_matches(img1, img2)
            num_inliers = len(matches)

            # 匹配数量，这里取最大值
            if num_inliers > max_score:
                max_score = num_inliers
                max_pair = [img1, img2]

    # 如果智能策略找不到，就退回到原来的逻辑
    if max_pair[0] is None:
        print("Warning: Smart init failed, falling back to max matches.")
        max_inliers = 0
        for i in range(len(image_ids)):
            img1 = image_ids[i]
            for img2 in scene_graph[img1]:
                if img1 >= img2: continue
                matches = load_matches(img1, img2)
                if len(matches) > max_inliers:
                    max_inliers = len(matches)
                    max_pair = [img1, img2]

    image_id1, image_id2 = sorted(max_pair)
    print(f"DEBUG: Smart Select initialized with: {image_id1} and {image_id2} (Matches: {max_score})")
    return image_id1, image_id2


def visualize_point_cloud(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw([pcd])


def load_matches(image_id1: str, image_id2: str) -> np.ndarray:
    """ Returns N x 2 indexes of matches """
    sorted_nodes = sorted([image_id1, image_id2])
    match_id = '_'.join(sorted_nodes)
    match_file = os.path.join(RANSAC_MATCH_DIR, match_id + '.npy')
    matches = np.load(match_file)
    if sorted_nodes[0] == image_id2:
        matches = np.flip(matches, axis=1)
    return matches


def get_init_extrinsics(image_id1: str, image_id2: str, intrinsics: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Assume that the image_id1 is at [I|0] and second image_id2 is at [R|t] where R, t are derived from the
    essential matrix.
    """
    extrinsics1 = np.zeros(shape=[3, 4], dtype=float)
    extrinsics1[:3, :3] = np.eye(3)

    match_id = '_'.join([image_id1, image_id2])
    essential_mtx_file = os.path.join(RANSAC_ESSENTIAL_DIR, match_id + '.npy')
    essential_mtx = np.load(essential_mtx_file)

    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points2d_1 = get_selected_points2d(image_id=image_id1, select_idxs=matches[:, 0])
    points2d_2 = get_selected_points2d(image_id=image_id2, select_idxs=matches[:, 1])

    extrinsics2 = np.zeros(shape=[3, 4], dtype=float)

    # --- IMPLEMENTATION START ---
    # Recover pose R and t from Essential Matrix
    # cv2.recoverPose returns the rotation and translation that transforms points from view 2 to view 1
    # We need the extrinsic matrix for view 2 (which maps world/view1 to view2)
    # Note: recoverPose assumes intrinsic matrix is provided
    _, R, t, mask = cv2.recoverPose(essential_mtx, points2d_1, points2d_2, intrinsics)

    # extrinsics2 = [R | t]
    extrinsics2[:3, :3] = R
    extrinsics2[:3, 3] = t.ravel()
    # --- IMPLEMENTATION END ---

    return extrinsics1, extrinsics2


def initialize(scene_graph: dict, intrinsics: np.ndarray):
    image_id1, image_id2 = get_init_image_ids(scene_graph)
    extrinsics1, extrinsics2 = get_init_extrinsics(image_id1=image_id1, image_id2=image_id2, intrinsics=intrinsics)
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    points3d = triangulate(image_id1=image_id1, image_id2=image_id2, extrinsics1=extrinsics1,
                           extrinsics2=extrinsics2, intrinsics=intrinsics, kp_idxs1=matches[:, 0],
                           kp_idxs2=matches[:, 1])

    num_matches = matches.shape[0]
    correspondences2d3d = {
        image_id1: {matches[i, 0]: i for i in range(num_matches)},
        image_id2: {matches[i, 1]: i for i in range(num_matches)}
    }
    return image_id1, image_id2, extrinsics1, extrinsics2, points3d, correspondences2d3d


def triangulate(image_id1: str, image_id2: str, kp_idxs1: np.ndarray, kp_idxs2: np.ndarray,
                extrinsics1: np.ndarray, extrinsics2: np.ndarray, intrinsics: np.ndarray):
    proj_pts1 = get_selected_points2d(image_id=image_id1, select_idxs=kp_idxs1)
    proj_pts2 = get_selected_points2d(image_id=image_id2, select_idxs=kp_idxs2)

    proj_mtx1 = np.matmul(intrinsics, extrinsics1)
    proj_mtx2 = np.matmul(intrinsics, extrinsics2)

    points3d = cv2.triangulatePoints(projMatr1=proj_mtx1, projMatr2=proj_mtx2,
                                     projPoints1=proj_pts1.transpose(1, 0), projPoints2=proj_pts2.transpose(1, 0))
    points3d = points3d.transpose(1, 0)
    points3d = points3d[:, :3] / points3d[:, 3].reshape(-1, 1)
    return points3d


def get_reprojection_residuals(points2d: np.ndarray, points3d: np.ndarray, intrinsics: np.ndarray,
                               rotation_mtx: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Projects the 3d points back into the image and computes the residuals.
    """
    residuals = np.zeros(points2d.shape[0])

    # --- IMPLEMENTATION START ---
    # Convert rotation matrix to rotation vector for cv2.projectPoints
    rvec, _ = cv2.Rodrigues(rotation_mtx)

    # Project 3D points to 2D image plane
    projected_points, _ = cv2.projectPoints(points3d, rvec, tvec, intrinsics, distCoeffs=None)
    projected_points = projected_points.reshape(-1, 2)

    # Compute Euclidean distance between observed and projected points
    residuals = np.linalg.norm(points2d - projected_points, axis=1)
    # --- IMPLEMENTATION END ---

    return residuals


def solve_pnp(image_id: str, point2d_idxs: np.ndarray, all_points3d: np.ndarray, point3d_idxs: np.ndarray,
              intrinsics: np.ndarray, num_ransac_iterations: int = 200, inlier_threshold: float = 5.0):
    """
    Solves the PnP problem using ransac.
    """
    num_pts = point2d_idxs.shape[0]
    assert num_pts >= 6, 'there should be at least 6 points'

    points2d = get_selected_points2d(image_id=image_id, select_idxs=point2d_idxs)
    points3d = all_points3d[point3d_idxs]

    has_valid_solution = False
    max_rotation_mtx, max_tvec, max_is_inlier, max_num_inliers = None, None, None, 0

    for _ in range(num_ransac_iterations):
        selected_idxs = np.random.choice(num_pts, size=6, replace=False).reshape(-1)
        selected_pts2d = points2d[selected_idxs, :]
        selected_pts3d = points3d[selected_idxs, :]

        # --- IMPLEMENTATION START ---
        # 1. call cv2.solvePnP
        # [修改] 使用 SOLVEPNP_EPNP，它比默认的 ITERATIVE 更稳健，不容易产生“压扁”或“一条线”的结果
        success, rvec, tvec = cv2.solvePnP(selected_pts3d, selected_pts2d, intrinsics, distCoeffs=None,
                                           flags=cv2.SOLVEPNP_EPNP)

        if not success:
            continue

        # 2. convert rotation vector to rotation matrix
        rotation_mtx, _ = cv2.Rodrigues(rvec)

        # 3. compute the reprojection residuals on ALL points (not just selected ones)
        residuals = get_reprojection_residuals(points2d, points3d, intrinsics, rotation_mtx, tvec)
        # --- IMPLEMENTATION END ---

        is_inlier = residuals <= inlier_threshold
        num_inliers = np.sum(is_inlier).item()

        if num_inliers > max_num_inliers:
            max_rotation_mtx = rotation_mtx
            max_tvec = tvec
            max_is_inlier = is_inlier
            max_num_inliers = num_inliers
            has_valid_solution = True

    assert has_valid_solution, "RANSAC PnP failed to find a valid solution"
    inlier_idxs = np.argwhere(max_is_inlier).reshape(-1)
    return max_rotation_mtx, max_tvec, inlier_idxs


def add_points3d(image_id1: str, image_id2: str, all_extrinsic: dict, intrinsics, points3d: np.ndarray,
                 correspondences2d3d: dict):
    """
    From the image pair (image_id1, image_id2), triangulate to get new points3d.
    """
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    # Find points in image2 that correspond to matches but are NOT yet in 3D correspondences
    points2d_idxs2 = np.setdiff1d(matches[:, 1], list(correspondences2d3d[image_id2].keys())).reshape(-1)

    if len(points2d_idxs2) == 0:
        return points3d, correspondences2d3d

    matches_idxs = np.array([np.argwhere(matches[:, 1] == i).reshape(-1)[0] for i in points2d_idxs2])
    matches = matches[matches_idxs, :]

    # --- IMPLEMENTATION START ---
    # triangulate between the image points for the unregistered matches
    kp_idxs1 = matches[:, 0]
    kp_idxs2 = matches[:, 1]

    new_points3d = triangulate(
        image_id1=image_id1,
        image_id2=image_id2,
        kp_idxs1=kp_idxs1,
        kp_idxs2=kp_idxs2,
        extrinsics1=all_extrinsic[image_id1],
        extrinsics2=all_extrinsic[image_id2],
        intrinsics=intrinsics
    )
    # --- IMPLEMENTATION END ---

    num_new_points3d = new_points3d.shape[0]
    new_points3d_idxs = np.arange(num_new_points3d) + points3d.shape[0]

    # Update correspondences
    # Initialize dict for image1 if keys don't exist, though typically passed dict should handle it
    if image_id1 not in correspondences2d3d:
        correspondences2d3d[image_id1] = {}

    for i in range(num_new_points3d):
        correspondences2d3d[image_id1][matches[i, 0]] = new_points3d_idxs[i]
        correspondences2d3d[image_id2][matches[i, 1]] = new_points3d_idxs[i]

    points3d = np.concatenate([points3d, new_points3d], axis=0)
    return points3d, correspondences2d3d


def get_next_pair(scene_graph: dict, registered_ids: list):
    """
    Finds the next match where the one of the images is unregistered while the other is registered.
    """
    max_new_id, max_registered_id, max_num_inliers = None, None, 0

    # --- IMPLEMENTATION START ---
    # Iterate over all currently registered images
    for reg_id in registered_ids:
        # Look at its neighbors in the scene graph
        neighbors = scene_graph.get(reg_id, [])
        for new_id in neighbors:
            # We are looking for an unregistered image
            if new_id not in registered_ids:
                matches = load_matches(reg_id, new_id)
                num_inliers = len(matches)

                if num_inliers > max_num_inliers:
                    max_num_inliers = num_inliers
                    max_new_id = new_id
                    max_registered_id = reg_id
    # --- IMPLEMENTATION END ---

    return max_new_id, max_registered_id


def get_pnp_2d3d_correspondences(image_id1: str, image_id2: str, correspondences2d3d: dict) -> (np.ndarray, np.ndarray):
    """
    Returns 2d and 3d correspondences for the image_id1 and the current world points.
    """
    matches = load_matches(image_id1=image_id1, image_id2=image_id2)
    # Find points in image2 that match points in image1 AND have existing 3D coordinates
    points2d_idxs2 = np.intersect1d(matches[:, 1], list(correspondences2d3d[image_id2].keys())).reshape(-1)

    match_idxs = [np.argwhere(matches[:, 1] == i).reshape(-1)[0] for i in points2d_idxs2]
    match_idxs = np.array(match_idxs)

    points2d_idxs1 = matches[match_idxs, 0]
    point3d_idxs = np.array([correspondences2d3d[image_id2][i] for i in points2d_idxs2])
    return points2d_idxs1, point3d_idxs


def bundle_adjustment(registered_ids: list, points3d: np.ndarray, correspondences2d3d: np.ndarray,
                      all_extrinsics: dict, intrinsics: np.ndarray, max_nfev: int = 30):
    # create parameters
    parameters = []
    for image_id in registered_ids:
        # convert rotation matrix to Rodriguez vector
        extrinsics = all_extrinsics[image_id]
        rotation_mtx = extrinsics[:3, :3]
        tvec = extrinsics[:, 3].reshape(3)
        rotation_mtx = rotation_mtx.astype(float)
        rvec, _ = cv2.Rodrigues(rotation_mtx)
        rvec = rvec.reshape(3)

        parameters.append(rvec)
        parameters.append(tvec)
    parameters.append(points3d.reshape(-1))
    parameters = np.concatenate(parameters, axis=0)

    # create correspondences
    points2d, camera_idxs, points3d_idxs = [], [], []
    for i, image_id in enumerate(registered_ids):
        correspondence_dict = correspondences2d3d[image_id]
        correspondences = np.array([[k, v] for k, v in correspondence_dict.items()])
        if correspondences.shape[0] == 0: continue
        pt2d_idxs = correspondences[:, 0]
        pt3d_idxs = correspondences[:, 1]

        pt2d = get_selected_points2d(image_id=image_id, select_idxs=pt2d_idxs)
        points2d.append(pt2d)
        points3d_idxs.append(pt3d_idxs)
        camera_idxs.append(np.ones(pt2d.shape[0]) * i)

    num_cameras = len(registered_ids)
    points2d = np.concatenate(points2d, axis=0)
    camera_idxs = np.concatenate(camera_idxs, axis=0).astype(int)
    points3d_idxs = np.concatenate(points3d_idxs, axis=0).astype(int)

    # run optimization
    results = least_squares(fun=compute_ba_residuals, x0=parameters, method='trf', max_nfev=max_nfev,
                            args=(intrinsics, num_cameras, points2d, camera_idxs, points3d_idxs), verbose=2)

    updated_parameters = results.x
    camera_parameters = updated_parameters[:num_cameras * 6]
    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    for i, image_id in enumerate(registered_ids):
        params = camera_parameters[i]
        rvec, tvec = params[:3], params[3:]
        rvec = rvec.reshape(1, 3)
        rotation_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics = np.concatenate([rotation_mtx, tvec.reshape(-1, 1)], axis=1)
        all_extrinsics[image_id] = extrinsics
    points3d = updated_parameters[num_cameras * 6:].reshape(-1, 3)
    return all_extrinsics, points3d


def incremental_sfm(registered_ids: list, all_extrinsic: dict, intrinsics: np.ndarray, points3d: np.ndarray,
                    correspondences2d3d: dict, scene_graph: dict, has_bundle_adjustment: bool) -> \
        (np.ndarray, dict, dict, list):
    all_image_ids = list(scene_graph.keys())
    num_steps = len(all_image_ids) - 2
    for _ in tqdm(range(num_steps)):
        # get pose for new image
        new_id, registered_id = get_next_pair(scene_graph=scene_graph, registered_ids=registered_ids)
        if new_id is None:
            print("Warning: Could not find any more images to register.")
            break

        points2d_idxs1, points3d_idxs = get_pnp_2d3d_correspondences(image_id1=new_id, image_id2=registered_id,
                                                                     correspondences2d3d=correspondences2d3d)
        # 如果点数不足 6 个，打印警告并停止重建，而不是崩溃
        if len(points2d_idxs1) < 6:
            print(
                f"Warning: Image {new_id} matches too few 3D points ({len(points2d_idxs1)}). Stopping reconstruction gracefully.")
            break
        rotation_mtx, tvec, inlier_idxs = solve_pnp(image_id=new_id, point2d_idxs=points2d_idxs1,
                                                    all_points3d=points3d, point3d_idxs=points3d_idxs,
                                                    intrinsics=intrinsics)

        # update correspondences
        new_extrinsics = np.concatenate([rotation_mtx, tvec.reshape(-1, 1)], axis=1)
        all_extrinsic[new_id] = new_extrinsics

        # Make sure dict exists
        if new_id not in correspondences2d3d:
            correspondences2d3d[new_id] = {}

        correspondences2d3d[new_id] = {points2d_idxs1[i]: points3d_idxs[i] for i in inlier_idxs}

        # create new points + update correspondences
        points3d, correspondences2d3d = add_points3d(image_id1=new_id, image_id2=registered_id,
                                                     all_extrinsic=all_extrinsic,
                                                     intrinsics=intrinsics, points3d=points3d,
                                                     correspondences2d3d=correspondences2d3d)
        registered_ids.append(new_id)

    if has_bundle_adjustment:
        all_extrinsic, points3d = bundle_adjustment(registered_ids=registered_ids, points3d=points3d,
                                                    all_extrinsics=all_extrinsic, intrinsics=intrinsics,
                                                    correspondences2d3d=correspondences2d3d)

    # assert len(np.setdiff1d(all_image_ids, registered_ids).reshape(-1)) == 0
    return points3d, all_extrinsic, correspondences2d3d, registered_ids


def main():
    # set seeds
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)

    with open(SCENE_GRAPH_FILE, 'r') as f:
        scene_graph = json.load(f)

    # run initialization step
    intrinsics = get_camera_intrinsics()
    image_id1, image_id2, extrinsic1, extrinsic2, points3d, correspondences2d3d = \
        initialize(scene_graph=scene_graph, intrinsics=intrinsics)
    registered_ids = [image_id1, image_id2]
    all_extrinsic = {
        image_id1: extrinsic1,
        image_id2: extrinsic2
    }

    points3d, all_extrinsic, correspondences2d3d, registered_ids = \
        incremental_sfm(registered_ids=registered_ids, all_extrinsic=all_extrinsic, intrinsics=intrinsics,
                        correspondences2d3d=correspondences2d3d, points3d=points3d, scene_graph=scene_graph,
                        has_bundle_adjustment=HAS_BUNDLE_ADJUSTMENT)

    os.makedirs(RESULT_DIR, exist_ok=True)
    points3d_save_file = os.path.join(RESULT_DIR, 'points3d.npy')
    np.save(points3d_save_file, points3d)

    correspondences2d3d = {a: {int(c): int(d) for c, d in b.items()} for a, b in correspondences2d3d.items()}
    correspondences2d3d_save_file = os.path.join(RESULT_DIR, 'correspondences2d3d.json')
    with open(correspondences2d3d_save_file, 'w') as f:
        json.dump(correspondences2d3d, f, indent=1)

    all_extrinsic = {image_id: [list(row) for row in extrinsic.astype(float)]
                     for image_id, extrinsic in all_extrinsic.items()}
    extrinsic_save_file = os.path.join(RESULT_DIR, 'all-extrinsic.json')
    with open(extrinsic_save_file, 'w') as f:
        json.dump(all_extrinsic, f, indent=1)

    registration_save_file = os.path.join(RESULT_DIR, 'registration-trajectory.txt')
    with open(registration_save_file, 'w') as f:
        for image_id in registered_ids:
            f.write(image_id + '\n')


if __name__ == '__main__':
    main()