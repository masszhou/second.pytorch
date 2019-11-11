import numpy as np
import torch
import argparse
from easydict import EasyDict as edict
import os
from second.protos import pipeline_pb2
from google.protobuf import text_format
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
import time
from second.utils.log_tool import SimpleModelLog
import psutil
import torchplus
import json


def argparser():
    parser = argparse.ArgumentParser(description="eval unet 3D")
    parser.add_argument("--config_path", default="./second/configs/car.fhd.config",
                        help="root path of lyft 3d object dataset")
    parser.add_argument("--model_dir", default="./tmp/",
                        help="path of saved weights")

    return parser.parse_args()


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])


def example_convert_to_torch(example, dtype=torch.float32, device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


if __name__ == "__main__":
    FLAGS = argparser()
    cfg = edict()
    cfg.config_path = FLAGS.config_path
    cfg.model_dir = FLAGS.model_dir
    os.makedirs(cfg.model_dir, exist_ok=True)
    cfg.multi_gpu = False
    cfg.measure_time = False
    cfg.display_step=50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # directly provide a config object. this usually used
    # when you want to train with several different parameters in
    # one script.
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(cfg.config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)

    config_file_bkp = "pipeline.config"
    with open(cfg.model_dir + "/" + config_file_bkp, "w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=False).to(device)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    print("num parameters:", len(list(net.parameters())))

    if cfg.multi_gpu:
        net_parallel = torch.nn.DataParallel(net)
    else:
        net_parallel = net

    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    fastai_optimizer = optimizer_builder.build(optimizer_cfg, net, mixed=False, loss_scale=loss_scale)

    if loss_scale < 0:
        loss_scale = "dynamic"

    amp_optimizer = fastai_optimizer

    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)

    float_dtype = torch.float32

    if cfg.multi_gpu:
        num_gpu = torch.cuda.device_count()
        print(f"MULTI-GPU: use {num_gpu} gpu")
        collate_fn = merge_second_batch_multigpu
    else:
        collate_fn = merge_second_batch
        num_gpu = 1

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=cfg.multi_gpu)
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu,
        shuffle=True,
        num_workers=input_cfg.preprocess.num_workers * num_gpu,
        pin_memory=False,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=not cfg.multi_gpu)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, # only support multi-gpu train
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(cfg.model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    try:
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()
            for example in dataloader:
                lr_scheduler.step(net.get_global_step())
                time_metrics = example["metrics"]
                example.pop("metrics")
                example_torch = example_convert_to_torch(example, float_dtype)

                batch_size = example["anchors"].shape[0]

                ret_dict = net_parallel(example_torch)
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"].mean()
                cls_neg_loss = ret_dict["cls_neg_loss"].mean()
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]

                cared = ret_dict["cared"]
                labels = example_torch["labels"]

                loss.backward()

                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step()
                amp_optimizer.zero_grad()
                net.update_global_step()
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if 'anchors_mask' not in example_torch:
                    num_anchors = example_torch['anchors'].shape[1]
                else:
                    num_anchors = int(example_torch['anchors_mask'][0].sum())
                global_step = net.get_global_step()

                if global_step % cfg.display_step == 0:

                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": np.mean(step_times),
                    }
                    metrics["runtime"].update(time_metrics[0])
                    step_times = []
                    metrics.update(net_metrics)
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())

                    metrics["misc"] = {
                        "num_vox": int(example_torch["voxels"].shape[0]),
                        "num_pos": int(num_pos),
                        "num_neg": int(num_neg),
                        "num_anchors": int(num_anchors),
                        "lr": float(amp_optimizer.lr),
                        "mem_usage": psutil.virtual_memory().percent,
                    }
                    model_logging.log_metrics(metrics, global_step)
                if global_step % steps_per_eval == 0:
                    torchplus.train.save_models(cfg.model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                step += 1
                if step >= total_step:
                    break
            if step >= total_step:
                break
    except Exception as e:
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models(cfg.model_dir, [net, amp_optimizer],
                                net.get_global_step())