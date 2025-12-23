import os
import numpy as np
from preprocess import RESULT_DIR
import open3d as o3d


def visualize_point_cloud(pts: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # [修改处] 改用 draw_geometries (旧版兼容模式)
    # 它可以避开中文路径导致的资源加载错误
    print("正在打开可视化窗口... (按 'H' 查看帮助，鼠标左键旋转，右键平移)")
    o3d.visualization.draw_geometries([pcd],
                                      window_name="SFM Result",
                                      width=800,
                                      height=600,
                                      left=50,
                                      top=50)


def main():
    points3d_save_file = os.path.join(RESULT_DIR, 'points3d.npy')

    if not os.path.exists(points3d_save_file):
        print(f"错误：找不到文件 {points3d_save_file}")
        return

    points3d = np.load(points3d_save_file)
    print(f"加载成功！共有 {points3d.shape[0]} 个 3D 点。")
    visualize_point_cloud(pts=points3d)


if __name__ == '__main__':
    main()