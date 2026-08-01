import os
import numpy as np
from PIL import Image
import scipy.io as scio

def generate_mock_dataset(base_dir="data_test"):
    scene_id = "scene_0000"
    camera = "realsense"
    robot = "leap_hand"

    # 1. Scenes
    scenes_path = os.path.join(base_dir, "scenes", scene_id, camera)
    os.makedirs(os.path.join(scenes_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scenes_path, "label"), exist_ok=True)
    os.makedirs(os.path.join(scenes_path, "meta"), exist_ok=True)

    # depth image (uint16 mm)
    depth_img = (np.ones((480, 640), dtype=np.uint16) * 1200)
    Image.fromarray(depth_img).save(os.path.join(scenes_path, "depth", "0000.png"))

    # label image (uint16)
    label_img = (np.ones((480, 640), dtype=np.uint16) * 1)
    Image.fromarray(label_img).save(os.path.join(scenes_path, "label", "0000.png"))

    # meta (.mat)
    intrinsic_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    meta = {
        "intrinsic_matrix": intrinsic_matrix,
        "factor_depth": np.array([[1000.0]], dtype=np.float32)
    }
    scio.savemat(os.path.join(scenes_path, "meta", "0000.mat"), meta)

    # camera poses & align mat
    camera_poses = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (256, 1, 1))
    np.save(os.path.join(scenes_path, "camera_poses.npy"), camera_poses)

    cam0_wrt_table = np.eye(4, dtype=np.float32)
    np.save(os.path.join(scenes_path, "cam0_wrt_table.npy"), cam0_wrt_table)

    # 2. Graspness
    graspness_path = os.path.join(base_dir, "dex_graspness_new", scene_id, camera)
    os.makedirs(graspness_path, exist_ok=True)
    graspness_scores = np.random.uniform(0.1, 0.9, size=(480 * 640,)).astype(np.float32)
    np.save(os.path.join(graspness_path, "0000.npy"), graspness_scores)

    # 3. Grasps
    grasps_path = os.path.join(base_dir, "dex_grasps_new", scene_id, robot)
    os.makedirs(grasps_path, exist_ok=True)

    # LEAP hand movable joint names from robot_models/urdf/leap_hand.urdf.
    joint_names = [
        "j12", "j13", "j14", "j15",
        "j0", "j1", "j2", "j3",
        "j8", "j9", "j10", "j11",
        "j4", "j5", "j6", "j7",
    ]

    sample_total = 64
    grasp_points = np.random.uniform(-0.05, 0.05, size=(sample_total, 3)).astype(np.float32)
    grasp_points[:, 2] = 1.2
    grasps_data = {
        "rotation": np.tile(np.eye(3, dtype=np.float32)[None, :, :], (sample_total, 1, 1)),
        "translation": grasp_points.copy(),
        "point": grasp_points,
    }

    for name in joint_names:
        grasps_data[name] = np.zeros((sample_total,), dtype=np.float32)

    np.savez(os.path.join(grasps_path, "grasp_0.npz"), **grasps_data)
    print(f"Synthetic dataset generated successfully at '{base_dir}'!")

if __name__ == "__main__":
    generate_mock_dataset()
