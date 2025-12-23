import numpy as np
import cv2


def compute_ba_residuals(parameters: np.ndarray, intrinsics: np.ndarray, num_cameras: int, points2d: np.ndarray,
                         camera_idxs: np.ndarray, points3d_idxs: np.ndarray) -> np.ndarray:
    """
    For each point2d in <points2d>, find its 3d point, reproject it back into the image and return the residual
    i.e. euclidean distance between the point2d and reprojected point.

    Args:
        parameters: list of camera parameters [r1, r2, r3, t1, t2, t3, ...] where r1, r2, r3 corresponds to the
                    Rodriguez vector. There are 6C + 3M parameters where C is the number of cameras
        intrinsics: camera intrinsics 3 x 3 array
        num_cameras: number of cameras, C
        points2d: N x 2 array of 2d points
        camera_idxs: camera_idxs[i] returns the index of the camera for points2d[i]
        points3d_idxs: points3d[points3d_idxs[i]] returns the 3d point corresponding to points2d[i]

    Returns:
        N residuals

    """
    num_camera_parameters = 6 * num_cameras
    camera_parameters = parameters[:num_camera_parameters]
    points3d = parameters[num_camera_parameters:]
    num_points3d = points3d.shape[0] // 3
    points3d = points3d.reshape(num_points3d, 3)

    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    camera_rvecs = camera_parameters[:, :3]
    camera_tvecs = camera_parameters[:, 3:]

    extrinsics = []
    for rvec in camera_rvecs:
        rot_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics.append(rot_mtx)
    extrinsics = np.array(extrinsics)  # C x 3 x 3
    extrinsics = np.concatenate([extrinsics, camera_tvecs.reshape(-1, 3, 1)], axis=2)  # C x 3 x 4

    residuals = np.zeros(shape=points2d.shape[0], dtype=float)
    """ 
    YOUR CODE HERE: 
    NOTE: DO NOT USE LOOPS 
    HINT: I used np.matmul; np.sum; np.sqrt; np.square, np.concatenate etc.
    """
    # 1. 准备数据：根据索引，将每个观测点对应的 3D 坐标提取出来
    # points3d 是 (M, 3), points3d_idxs 是 (N,) -> 得到 (N, 3)
    # 这意味着第 i 行是第 i 个观测点对应的 3D 坐标
    current_points3d = points3d[points3d_idxs]

    # 2. 转换为齐次坐标 (Homogeneous Coordinates)
    # 在 (N, 3) 后面拼上一列 1，变成 (N, 4)
    ones = np.ones((current_points3d.shape[0], 1))
    current_points3d_homog = np.concatenate([current_points3d, ones], axis=1)

    # 3. 准备相机参数：根据索引，提取每个观测点对应的相机外参
    # extrinsics 是 (C, 3, 4), camera_idxs 是 (N,) -> 得到 (N, 3, 4)
    current_extrinsics = extrinsics[camera_idxs]

    # 4. 世界坐标系 -> 相机坐标系 (批量矩阵乘法)
    # 形状变换: (N, 3, 4) 乘以 (N, 4, 1) -> 得到 (N, 3, 1)
    # P_cam = R * P_world + t
    points_camera_frame = np.matmul(current_extrinsics, current_points3d_homog.reshape(-1, 4, 1))

    # 5. 相机坐标系 -> 像素坐标系 (投影)
    # 形状变换: (3, 3) 乘以 (N, 3, 1) -> 得到 (N, 3, 1)
    # P_img_homog = K * P_cam
    # 注意：intrinsics 会自动广播(broadcast)到 N 个样本上
    points_image_homog = np.matmul(intrinsics, points_camera_frame)

    # 把它变回 (N, 3) 以便操作
    points_image_homog = points_image_homog.reshape(-1, 3)

    # 6. 透视除法 (Perspective Division)
    # 将齐次坐标 (u', v', w') 转换为笛卡尔坐标 (u, v)
    # u = u' / w', v = v' / w'
    # 为了数值稳定，通常除数可以加个极小值 1e-6 (虽然在这个作业里一般不需要)
    z = points_image_homog[:, 2:3]
    points_projected = points_image_homog[:, :2] / z

    # 7. 计算残差 (欧氏距离)
    # 观测点 points2d - 投影点 points_projected
    diff = points2d - points_projected

    # 计算每一行的向量长度 (L2 Norm)
    # residuals[i] = sqrt(dx^2 + dy^2)
    residuals = np.sqrt(np.sum(np.square(diff), axis=1))

    
    """ END YOUR CODE HERE """
    return residuals
