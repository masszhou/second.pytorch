import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool
import argparse
import pptk


def argparser():
    parser = argparse.ArgumentParser(description="eval unet 3D")
    parser.add_argument("--config_path", default="./second/configs/car.lite.config",
                        help="root path of lyft 3d object dataset")
    parser.add_argument("--ckpt_path", default="./pretrained_models_v1.5/car_lite/voxelnet-15500.tckpt",
                        help="path of saved weights")
    parser.add_argument("--batch_size", default=1,
                        help="batch size")

    return parser.parse_args()


if __name__ == "__main__":
    FLAGS = argparser()

    # ======================================================
    # Read Config file
    # ======================================================
    config_path = FLAGS.config_path
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    config_tool.change_detection_range_v2(model_cfg, [-50, -50, 50, 50])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================================================
    # Build Network, Target Assigner and Voxel Generator
    # ======================================================
    ckpt_path = FLAGS.ckpt_path
    net = build_network(model_cfg).to(device).eval()
    net.load_state_dict(torch.load(ckpt_path))
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    # ======================================================
    # Generate Anchors
    # ======================================================
    grid_size = voxel_generator.grid_size
    feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
    feature_map_size = [*feature_map_size, 1][::-1]

    anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
    anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
    anchors = anchors.view(1, -1, 7)

    # ======================================================
    # Read KITTI infos
    # ======================================================
    info_path = input_cfg.dataset.kitti_info_path
    root_path = Path(input_cfg.dataset.kitti_root_path)
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    # ======================================================
    # Load Point Cloud, Generate Voxels
    # ======================================================
    info = infos[564]
    v_path = info["point_cloud"]['velodyne_path']
    v_path = str(root_path / v_path)

    # # https://www.nuscenes.org/data-collection
    # v_path = "/media/zzhou/data/data-lyft-3D-objects/lidar/host-a007_lidar1_1230678335302050856.bin"
    # points_lyft = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 5])
    # points = np.zeros([points_lyft.shape[0], 4], dtype=np.float32)
    # points[:, 0] = points_lyft[:, 0]
    # points[:, 1] = points_lyft[:, 1]
    # points[:, 2] = points_lyft[:, 2]
    # points[:, 3] = points_lyft[:, 3]

    # v_path = "/media/zzhou/data/data-KITTI/object/testing/velodyne/000010.bin"
    # points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 4])

    v_path = "/media/zzhou/data/data-lyft-3D-objects/train_kitti/velodyne/ada2945b41b1c32054f86fe1bd488bd593c1be35fc4e35c65e40d422f5ae4f52.bin"
    points_lyft = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 4])
    points = np.zeros([points_lyft.shape[0], 4], dtype=np.float32)
    points[:, 0] = points_lyft[:, 1] * (-1)
    points[:, 1] = points_lyft[:, 0]
    points[:, 2] = points_lyft[:, 2]
    points[:, 3] = points_lyft[:, 3]

    print("v_path = ", v_path)
    print(points.shape)
    print(np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2]), np.max(points[:, 3]))
    print(np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]), np.min(points[:, 3]))

    # pptk viewer from HERE
    v = pptk.viewer(points[:, :3], points[:, 3])
    v.set(point_size=0.01, lookat=(0.0, 0.0, 0.0))

    res = voxel_generator.generate(points[:, :4], max_voxels=90000)
    voxels, coords, num_points = res['voxels'], res['coordinates'], res['num_points_per_voxel']
    print("voxels.shape = ", voxels.shape)
    # add batch idx to coords
    coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
    coords = torch.tensor(coords, dtype=torch.int32, device=device)
    num_points = torch.tensor(num_points, dtype=torch.int32, device=device)

    # ======================================================
    # Detection
    # ======================================================
    example = {
        "anchors": anchors,
        "voxels": voxels,
        "num_points": num_points,
        "coordinates": coords,
    }
    pred = net(example)[0]

    # ======================================================
    # Simple Vis
    # ======================================================
    boxes_lidar = pred['box3d_lidar'].detach().cpu().numpy()  # I guess (x, y, z, w, l, h, yaw)
    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["label_preds"].detach().cpu().numpy()
    print(boxes_lidar.shape)

    vis_voxel_size = [0.1, 0.1, 0.1]
    vis_point_range = [-50, -30, -3, 50, 30, 1]
    bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
    bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
    plt.imshow(bev_map)
    plt.show()