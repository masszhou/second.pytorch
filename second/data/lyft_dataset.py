from lyft_dataset_sdk.lyftdataset import LyftDataset
from datetime import datetime
import pandas as pd
import numpy as np
from pyquaternion import Quaternion
import pickle
from pathlib import Path


lyft_name_mapping = {"car": 0,
                     "other_vehicle": 1,
                     "pedestrian": 2,
                     "bicycle": 3,
                     "truck": 4,
                     "bus": 5,
                     "motorcycle": 6,
                     "animal": 7,
                     "emergency_vehicle": 8}


def _scan_sample_token(level5data,
                       df,
                       test=False):
    dataset_infos = []
    # each info is a dict, contains following information
    # "lidar_path": lidar_path,
    # "cam_front_path": cam_path,
    # "token": sample["token"],
    # "sweeps": [],
    # "lidar2ego_translation": calib_record['translation'],
    # "lidar2ego_rotation": calib_record['rotation'],
    # "ego2global_translation": pose_record['translation'],
    # "ego2global_rotation": pose_record['rotation'],
    # "timestamp": sample["timestamp"],
    # "gt_boxes" = gt_boxes
    # "gt_names" = names
    # "gt_velocity" = velocity.reshape(-1, 2)
    # "num_lidar_pts" = np.array([a["num_lidar_pts"] for a in annotations])
    # "num_radar_pts" = np.array([a["num_radar_pts"] for a in annotations])

    for first_sample_token in df.first_sample_token.values:
        sample_token = first_sample_token
        while sample_token:
            sample = level5data.get("sample", sample_token)
            lidar_top_token = sample["data"]["LIDAR_TOP"]
            # lidar_path = level5data.get_sample_data_path(lidar_top_token)
            lidar_path, annotations_boxes, _ = level5data.get_sample_data(lidar_top_token)
            # return a tuple
            # [0] is lidar_path
            # [1] is an annotation list of lyft_dataset_sdk.utils.data_classes.Box
            # content of lyft_dataset_sdk.utils.data_classes.Box looks like
            #   label: nan,
            #   score: nan,
            #   xyz: [-30.15, 54.87, 0.22],
            #   wlh: [2.04, 4.96, 1.56],
            #   rot axis: [0.05, -0.02, -1.00],
            #   ang(degrees): 86.75,
            #   ang(rad): 1.51,
            #   vel: nan, nan, nan,
            #   name: car,
            #   token: 2fe740def816eed5c8d4f02e689b2bd7a6d1897fc282873b2053e81620ef091c

            lidar_record = level5data.get('sample_data', sample['data']["LIDAR_TOP"])
            # ==== lidar_data looks like
            # {'is_key_frame': True,
            #  'prev': '',
            #  'fileformat': 'bin',
            #  'ego_pose_token': 'ad84a94d701323a49fd890e31553c3a60405e9428b84749c9f3f85226829fbcd',
            #  'timestamp': 1547166237401681.8,
            #  'next': 'b1e97f7040470a2171ed110bfe0e0d5dfe7a6b5ed92ed53e8d14ecb4140dd298',
            #  'token': '5d6d78ab8d58702d82284d89c164311b79dbf131be93f477f0cb2fcfe673ba74',
            #  'sample_token': 'cea0bba4b425537cca52b17bf81569a20da1ca6d359f33227f0230d59d9d2881',
            #  'filename': 'lidar/host-a005_lidar1_1231201437401681856.bin',
            #  'calibrated_sensor_token': 'c87689b89f1bfa3e3b964be42bf25bd27298f90696d4da851aacb2a51db68acf',
            #  'sensor_modality': 'lidar',
            #  'channel': 'LIDAR_TOP'}

            calib_record = level5data.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
            # ==== cali_data looks like, rotation is Quaternion
            # {'sensor_token': '25bf751d7e35f295393d8a418731474b21c1f702e878c4553f112397caa48c08',
            #  'rotation': [0.020725150906434158,
            #               -0.011341959829999996,
            #               0.000235129466046585,
            #               0.9997208474275479],
            #  'camera_intrinsic': [],
            #  'translation': [1.1970985163266379,
            #                  0.00012035268047430447,
            #                  1.8279670539933053],
            #  'token': 'c87689b89f1bfa3e3b964be42bf25bd27298f90696d4da851aacb2a51db68acf'}

            pose_record = level5data.get('ego_pose', lidar_record['ego_pose_token'])
            # ==== pose_record looks like, rotation is Quaternion
            # {'rotation': [0.16443143502531854,
            #               0.02522225267574294,
            #               0.0117456840556274,
            #               0.9859960345009208],
            #  'translation': [871.9098767426335, 2567.7695447600618, -21.791296281900813],
            #  'token': 'ad84a94d701323a49fd890e31553c3a60405e9428b84749c9f3f85226829fbcd',
            #  'timestamp': 1547166237401681.8}

            cam_front_token = sample["data"]["CAM_FRONT"]
            cam_path = level5data.get_sample_data_path(cam_front_token)

            info = {
                "lidar_path": lidar_path,
                "cam_front_path": cam_path,
                "token": sample["token"],
                "sweeps": [],
                "lidar2ego_translation": calib_record['translation'],
                "lidar2ego_rotation": calib_record['rotation'],
                "ego2global_translation": pose_record['translation'],
                "ego2global_rotation": pose_record['rotation'],
                "timestamp": sample["timestamp"],
            }

            l2e_r = info["lidar2ego_rotation"]
            l2e_t = info["lidar2ego_translation"]
            e2g_r = info["ego2global_rotation"]
            e2g_t = info["ego2global_translation"]
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            if not test:
                annotations = [level5data.get('sample_annotation', token) for token in sample['anns']]
                # annotations looks like
                # [{'token': '2fe740def816eed5c8d4f02e689b2bd7a6d1897fc282873b2053e81620ef091c',
                #   'num_lidar_pts': -1,
                #   'size': [2.043, 4.958, 1.561],
                #   'sample_token': 'cea0bba4b425537cca52b17bf81569a20da1ca6d359f33227f0230d59d9d2881',
                #   'rotation': [-0.5878361278667221, 0, 0, 0.8089800286624255],
                #   'prev': '',
                #   'translation': [862.778784827005, 2630.267585970499, -19.497431586665336],
                #   'num_radar_pts': 0,
                #   'attribute_tokens': ['1ba8c9a8bda54423fa710b0af1441d849ecca8ed7b7f9393ba1794afe4aa6aa2',
                #                        'daf16a3f6499553cc5e1df4a456de5ee46e2e6b06544686d918dfb1ddb088f6f'],
                #   'next': '0b6e9b1da777d5ad343cdca7879e7ee7c2ac4903dabf206bc1cfffca1a286ce5',
                #   'instance_token': '26f4cd03ff22844f80e2eb0c24e20f87f4307a31043671f7d2b97425f415bb19',
                #   'visibility_token': '',
                #   'category_name': 'car'},
                #   ... ]

                # lyft_dataset_sdk.utils.data_classes.Box API
                locs = np.array([b.center for b in annotations_boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in annotations_boxes]).reshape(-1, 3)
                rots = np.array([b.orientation.yaw_pitch_roll[0] for b in annotations_boxes]).reshape(-1, 1)

                velocity = np.array(
                    [level5data.box_velocity(token)[:2] for token in sample['anns']])
                # convert velo from global to lidar
                for i in range(len(annotations_boxes)):
                    velo = np.array([*velocity[i], 0.0])
                    velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                        l2e_r_mat).T
                    velocity[i] = velo[:2]

                names = [b.name for b in annotations_boxes]
                # for i in range(len(names)):
                #     if names[i] in lyft_name_mapping:
                #         names[i] = lyft_name_mapping[names[i]]
                names = np.array(names)

                # we need to convert rot to SECOND format.
                # change the rot format will break all checkpoint, so...
                gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
                assert len(gt_boxes) == len(
                    annotations), f"{len(gt_boxes)}, {len(annotations)}"
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = names
                info["gt_velocity"] = velocity.reshape(-1, 2)
                info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
                info["num_radar_pts"] = np.array([a["num_radar_pts"] for a in annotations])
            dataset_infos.append(info)
    return dataset_infos


def create_lyft_infos(root_path=Path("/media/zzhou/data/data-lyft-3D-objects/")):
    # ====================================================================
    # A. Creating an index and splitting into train and validation scenes
    # ====================================================================
    level5data = LyftDataset(data_path=root_path, json_path=root_path + 'train_data', verbose=True)

    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    records = [(level5data.get('sample', record['first_sample_token'])['timestamp'], record) for record in level5data.scene]
    entries = []
    for start_time, record in sorted(records):
        start_time = level5data.get('sample', record['first_sample_token'])['timestamp'] / 1000000
        token = record['token']
        name = record['name']
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record['name'].split("-")[:2])
        first_sample_token = record["first_sample_token"]
        entries.append((host, name, date, token, first_sample_token))
    df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])

    validation_hosts = ["host-a007", "host-a008", "host-a009"]

    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]
    print(len(train_df), len(validation_df), "train/validation split scene counts")

    # ====================================================================
    # B. build SECOND database
    # ====================================================================
    metadata = {
        "version": f"{len(train_df)} train / {len(validation_df)} validation split scene counts",
    }

    train_lyft_infos = _scan_sample_token(level5data, train_df)
    data = {
        "infos": train_lyft_infos,
        "metadata": metadata,
    }
    with open(root_path / "infos_train.pkl", 'wb') as f:
        pickle.dump(data, f)

    val_lyft_infos = _scan_sample_token(level5data, validation_df)
    data["infos"] = val_lyft_infos
    with open(root_path / "infos_val.pkl", 'wb') as f:
        pickle.dump(data, f)