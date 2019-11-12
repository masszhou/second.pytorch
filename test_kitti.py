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

    return parser.parse_args()


class SecondDetector:
    def __init__(self, config_filepath, weight_filepath):
        # ======================================================
        # Read Config file
        # ======================================================
        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_filepath, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        self.input_cfg = self.config.eval_input_reader
        self.model_cfg = self.config.model.second
        config_tool.change_detection_range_v2(self.model_cfg, [-50, -50, 50, 50])

        # ======================================================
        # Build Network, Target Assigner and Voxel Generator
        # ======================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(self.model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(weight_filepath))

        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator

        # ======================================================
        # Generate Anchors
        # ======================================================
        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(self.model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]

        self.anchors = self.target_assigner.generate_anchors(feature_map_size)["anchors"]
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)
        self.anchors = self.anchors.view(1, -1, 7)

    @staticmethod
    def load_pts_from_file(pts_filepath, pts_dim=4):
        pts = np.fromfile(pts_filepath, dtype=np.float32, count=-1).reshape([-1, pts_dim])
        return pts

    def load_example_from_points(self, points):
        res = self.voxel_generator.generate(points[:, :4], max_voxels=90000)
        voxels, coords, num_points = res['voxels'], res['coordinates'], res['num_points_per_voxel']
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        return {
            'anchors': self.anchors,
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coords,
        }

    def predict_on_points(self, pts_array):
        example = self.load_example_from_points(pts_array)
        pred = self.net(example)[0]
        boxes_lidar = pred['box3d_lidar'].detach().cpu().numpy()  # I guess (x, y, z, w, l, h, yaw)
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["label_preds"].detach().cpu().numpy()

        return boxes_lidar, scores, labels

    @staticmethod
    def visualize_bev(points, boxes_lidar):
        vis_voxel_size = [0.1, 0.1, 0.1]
        vis_point_range = [-50, -30, -3, 50, 30, 1]
        bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
        bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, boxes_lidar, [0, 255, 0], 2)
        plt.imshow(bev_map)
        plt.show()


if __name__ == "__main__":
    FLAGS = argparser()

    detector = SecondDetector(FLAGS.config_path, FLAGS.ckpt_path)

    # # https://www.nuscenes.org/data-collection
    # v_path = "/media/zzhou/data/data-lyft-3D-objects/lidar/host-a007_lidar1_1230678335302050856.bin"
    # points_lyft = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 5])
    # points = np.zeros([points_lyft.shape[0], 4], dtype=np.float32)
    # points[:, 0] = points_lyft[:, 0]
    # points[:, 1] = points_lyft[:, 1]
    # points[:, 2] = points_lyft[:, 2]
    # points[:, 3] = points_lyft[:, 3]

    v_path = "/media/zzhou/data/data-KITTI/object/testing/velodyne/000050.bin"
    points = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 4])

    # v_path = "/media/zzhou/data/data-lyft-3D-objects/train_kitti/velodyne/ada2945b41b1c32054f86fe1bd488bd593c1be35fc4e35c65e40d422f5ae4f52.bin"
    # points_lyft = np.fromfile(v_path, dtype=np.float32, count=-1).reshape([-1, 4])
    # points = np.zeros([points_lyft.shape[0], 4], dtype=np.float32)
    # points[:, 0] = points_lyft[:, 1] * (-1)
    # points[:, 1] = points_lyft[:, 0]
    # points[:, 2] = points_lyft[:, 2]
    # points[:, 3] = points_lyft[:, 3]
    # x_mask = (points[:, 0] > -70) * (points[:, 0] < 70)
    # y_mask = (points[:, 1] > -70) * (points[:, 1] < 70)
    # mask = x_mask * y_mask
    # points = points[mask]

    boxes_lidar, scores, labels = detector.predict_on_points(points)
    print(boxes_lidar)
    print(scores)
    print(labels)

    detector.visualize_bev(points, boxes_lidar)