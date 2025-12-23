import os
import numpy as np
import cv2
import pickle as pkl
from tqdm import tqdm
import networkx as nx
import shutil
import json
import torch.utils.data as tdata
import argparse

# --- 路径设置 ---
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
PREDICTION_DIR = os.path.join(PROJECT_DIR, 'predictions')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, choices=['temple', 'mini-temple'], default='temple')
argparser.add_argument('--ba', action='store_true')
args = argparser.parse_args()

DATASET = args.dataset
DATASET_DIR = os.path.join(DATA_DIR, DATASET)
IMAGE_DIR = os.path.join(DATASET_DIR, 'images')
INTRINSICS_FILE = os.path.join(DATASET_DIR, 'intrinsics.txt')

SAVE_DIR = os.path.join(PREDICTION_DIR, DATASET)
BAD_MATCHES_FILE = os.path.join(SAVE_DIR, 'bad-match.txt')
KEYPOINT_DIR = os.path.join(SAVE_DIR, 'keypoints')
BF_MATCH_DIR = os.path.join(SAVE_DIR, 'bf-match')
BF_MATCH_IMAGE_DIR = os.path.join(SAVE_DIR, 'bf-match-images')

RANSAC_MATCH_DIR = os.path.join(SAVE_DIR, 'ransac-match')
RANSAC_ESSENTIAL_DIR = os.path.join(SAVE_DIR, 'ransac-fundamental')
RANSAC_MATCH_IMAGE_DIR = os.path.join(SAVE_DIR, 'ransac-match-images')
BAD_RANSAC_MATCHES_FILE = os.path.join(SAVE_DIR, 'bad-ransac-matches.txt')
SCENE_GRAPH_FILE = os.path.join(SAVE_DIR, 'scene-graph.json')

HAS_BUNDLE_ADJUSTMENT = args.ba
SPLIT = 'bundle-adjustment' if HAS_BUNDLE_ADJUSTMENT else 'no-bundle-adjustment'
RESULT_DIR = os.path.join(SAVE_DIR, 'results', SPLIT)


class ParallelDataset(tdata.Dataset):
    def __init__(self, data: list, func):
        super(ParallelDataset, self).__init__()
        self.data = data
        self.func = func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        out = self.func(*data)
        return out


def get_camera_intrinsics() -> np.ndarray:
    with open(INTRINSICS_FILE, 'r') as f:
        intrinsics = f.readlines()
    intrinsics = [line.strip().split(' ') for line in intrinsics]
    intrinsics = np.array(intrinsics).astype(float)
    return intrinsics


def encode_keypoint(kp: cv2.KeyPoint) -> tuple:
    return kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id


def decode_keypoint(kp: tuple) -> cv2.KeyPoint:
    return cv2.KeyPoint(x=kp[0][0], y=kp[0][1], size=kp[1], angle=kp[2], response=kp[3],
                        octave=kp[4], class_id=kp[5])


def get_detected_keypoints(image_id: str) -> (list, list):
    keypoint_file = os.path.join(KEYPOINT_DIR, image_id + '.pkl')
    with open(keypoint_file, 'rb') as _f:
        keypoint = pkl.load(_f)
    keypoints, descriptors = keypoint['keypoints'], keypoint['descriptors']
    keypoints = [decode_keypoint(_kp) for _kp in keypoints]
    return keypoints, descriptors


def parallel_processing(data: list, func, batchsize: int = 1, shuffle: bool = False, num_workers: int = 0):
    # 注意：Windows下多进程(num_workers>0)容易报错，这里强制设为0以保证稳定运行
    dataset = ParallelDataset(data=data, func=func)
    dataloader = tdata.DataLoader(dataset=dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batchsize)
    out = []
    for batch_out in tqdm(dataloader):
        out.extend(list(batch_out))
    return out


# --- 任务 1: 检测关键点 (Assignment Task: detect_keypoints) ---
def detect_keypoints(image_file: str):
    image_id = os.path.basename(image_file)[:-4]
    save_file = os.path.join(KEYPOINT_DIR, image_id + '.pkl')

    keypoints, descriptors = [], []

    # [FIXED CODE START]
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_file}")
        return image_id

    try:
        # 尝试新版 OpenCV 写法 (4.5.x+)
        sift = cv2.SIFT_create()
    except AttributeError:
        # 兼容旧版 OpenCV 写法
        sift = cv2.xfeatures2d.SIFT_create()

    # 检测关键点和计算描述符 [cite: 55]
    keypoints, descriptors = sift.detectAndCompute(img, None)
    # [FIXED CODE END]

    keypoints_encoded = [encode_keypoint(kp=kp) for kp in keypoints]
    save_dict = {
        'keypoints': keypoints_encoded,
        'descriptors': descriptors
    }

    with open(save_file, 'wb') as f:
        pkl.dump(save_dict, f)
    return image_id


# --- 任务 2: 特征匹配 (Assignment Task: create_feature_matches) ---
def create_feature_matches(image_file1: str, image_file2: str, lowe_ratio: float = 0.75, min_matches: int = 10):
    image_id1 = os.path.basename(image_file1)[:-4]
    image_id2 = os.path.basename(image_file2)[:-4]
    match_id = '{}_{}'.format(image_id1, image_id2)

    match_save_file = os.path.join(BF_MATCH_DIR, match_id + '.npy')
    image_save_file = os.path.join(BF_MATCH_IMAGE_DIR, match_id + '.png')

    keypoints1, descriptors1 = get_detected_keypoints(image_id=image_id1)
    keypoints2, descriptors2 = get_detected_keypoints(image_id=image_id2)

    good_matches = []

    # [FIXED CODE START]
    if descriptors1 is not None and descriptors2 is not None and len(descriptors1) > 0 and len(descriptors2) > 0:
        # 使用 BFMatcher 进行 KNN 匹配 [cite: 56]
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        # 使用 Lowe's Ratio Test 过滤匹配点
        for m, n in matches:
            if m.distance < lowe_ratio * n.distance:
                good_matches.append([m])
    # [FIXED CODE END]

    if len(good_matches) < min_matches:
        return match_id

    # 可视化
    image1 = cv2.imread(image_file1)
    image2 = cv2.imread(image_file2)
    save_image = cv2.drawMatchesKnn(img1=image1, keypoints1=keypoints1, img2=image2, keypoints2=keypoints2,
                                    matches1to2=good_matches, outImg=None,
                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(image_save_file, save_image)

    good_matches_flat = [match[0] for match in good_matches]
    feature_matches = []
    for match in good_matches_flat:
        feature_matches.append([match.queryIdx, match.trainIdx])
    feature_matches = np.array(feature_matches)
    np.save(match_save_file, feature_matches)
    return match_id


def get_selected_points2d(image_id: str, select_idxs: np.ndarray) -> np.ndarray:
    keypoints, _ = get_detected_keypoints(image_id=image_id)
    points2d = [keypoints[i].pt for i in select_idxs]
    points2d = np.array(points2d)
    return points2d


#  几何验证
def create_ransac_matches(image_file1: str, image_file2: str,
                          min_feature_matches: int = 15, ransac_threshold: float = 1.0):
    image_id1 = os.path.basename(image_file1)[:-4]
    image_id2 = os.path.basename(image_file2)[:-4]
    match_id = '{}_{}'.format(image_id1, image_id2)

    match_save_file = os.path.join(RANSAC_MATCH_DIR, match_id + '.npy')
    essential_mtx_save_file = os.path.join(RANSAC_ESSENTIAL_DIR, match_id + '.npy')
    image_save_file = os.path.join(RANSAC_MATCH_IMAGE_DIR, match_id + '.png')
    feature_match_file = os.path.join(BF_MATCH_DIR, match_id + '.npy')

    if not os.path.exists(feature_match_file):
        return match_id

    match_idxs = np.load(feature_match_file)
    if match_idxs.shape[0] < min_feature_matches:
        return match_id

    points1 = get_selected_points2d(image_id=image_id1, select_idxs=match_idxs[:, 0])
    points2 = get_selected_points2d(image_id=image_id2, select_idxs=match_idxs[:, 1])
    camera_intrinsics = get_camera_intrinsics()

    is_inlier = np.zeros(shape=points1.shape[0], dtype=bool)
    essential_mtx = np.eye(3)

    # [FIXED CODE START]
    # 使用 findEssentialMatrix 和 RANSAC 进行几何验证 [cite: 69]
    if len(points1) >= 5:  # 至少需要5个点
        essential_mtx, mask = cv2.findEssentialMat(points1, points2, camera_intrinsics,
                                                      method=cv2.RANSAC,
                                                      prob=0.999,
                                                      threshold=ransac_threshold)
        if mask is not None:
            is_inlier = mask.ravel().astype(bool)
    # [FIXED CODE END]

    inlier_idxs = np.argwhere(is_inlier).reshape(-1)
    if len(inlier_idxs) == 0:
        return match_id

    inlier_match_idxs = match_idxs[inlier_idxs, :]
    np.save(match_save_file, inlier_match_idxs)
    np.save(essential_mtx_save_file, essential_mtx)

    # 可视化
    image1 = cv2.imread(image_file1)
    image2 = cv2.imread(image_file2)
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    vis_h = max(h1, h2)
    vis_w1 = int(w1 * vis_h / h1)
    vis_w2 = int(w2 * vis_h / h2)
    image1 = cv2.resize(image1, (vis_w1, vis_h))
    image2 = cv2.resize(image2, (vis_w2, vis_h))

    save_image = np.concatenate([image1, image2], axis=1)
    offset = image1.shape[1]

    match_pts = np.concatenate([points1[is_inlier], points2[is_inlier]], axis=1)
    match_pts = match_pts.astype(int)
    for x1, y1, x2, y2 in match_pts:
        cv2.line(img=save_image, pt1=(x1, y1), pt2=(x2 + offset, y2), thickness=1, color=(0, 255, 0))
    cv2.imwrite(image_save_file, save_image)
    return match_id


#  构建场景图
def create_scene_graph(image_files: list, min_num_inliers: int = 15):
    graph = nx.Graph()
    image_ids = [os.path.basename(file)[:-4] for file in image_files]
    graph.add_nodes_from(list(range(len(image_files))))

    # [FIXED CODE START]
    # 如果两张图之间有足够的几何验证内点，则添加一条边 [cite: 58, 59]
    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            id1 = image_ids[i]
            id2 = image_ids[j]
            match_file = os.path.join(RANSAC_MATCH_DIR, f'{id1}_{id2}.npy')

            if os.path.exists(match_file):
                matches = np.load(match_file)
                if len(matches) >= min_num_inliers:
                    # 使用索引作为节点添加边
                    graph.add_edge(i, j)
    # [FIXED CODE END]

    graph_dict = {node: [] for node in image_ids}
    for i1, i2 in graph.edges:
        node1 = image_ids[i1]
        node2 = image_ids[i2]
        graph_dict[node1].append(node2)
        graph_dict[node2].append(node1)

    graph_dict = {node: list(np.unique(neighbors).reshape(-1)) for node, neighbors in graph_dict.items()}
    with open(SCENE_GRAPH_FILE, 'w') as f:
        json.dump(graph_dict, f, indent=4)


def main():
    image_files = [os.path.join(IMAGE_DIR, filename) for filename in sorted(os.listdir(IMAGE_DIR))]
    if len(image_files) == 0:
        print(f"Error: No images found in {IMAGE_DIR}")
        return

    print('INFO: detecting image keypoints...')
    shutil.rmtree(KEYPOINT_DIR, ignore_errors=True)
    os.makedirs(KEYPOINT_DIR, exist_ok=True)
    parallel_processing(data=[(file,) for file in image_files], func=detect_keypoints)

    print('INFO: creating pairwise matches between images...')
    matches = []
    for i, file1 in enumerate(image_files):
        for file2 in image_files[i + 1:]:
            matches.append((file1, file2))

    shutil.rmtree(BF_MATCH_DIR, ignore_errors=True)
    shutil.rmtree(BF_MATCH_IMAGE_DIR, ignore_errors=True)
    os.makedirs(BF_MATCH_DIR, exist_ok=True)
    os.makedirs(BF_MATCH_IMAGE_DIR, exist_ok=True)
    parallel_processing(data=matches, func=create_feature_matches)

    print('INFO: creating ransac matches...')
    shutil.rmtree(RANSAC_MATCH_DIR, ignore_errors=True)
    shutil.rmtree(RANSAC_MATCH_IMAGE_DIR, ignore_errors=True)
    shutil.rmtree(RANSAC_ESSENTIAL_DIR, ignore_errors=True)
    os.makedirs(RANSAC_MATCH_DIR, exist_ok=True)
    os.makedirs(RANSAC_MATCH_IMAGE_DIR, exist_ok=True)
    os.makedirs(RANSAC_ESSENTIAL_DIR, exist_ok=True)
    parallel_processing(data=matches, func=create_ransac_matches)

    print('INFO: creating scene graph...')
    create_scene_graph(image_files=image_files)


if __name__ == '__main__':
    main()