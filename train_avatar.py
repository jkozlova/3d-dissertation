import copy
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torchvision
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import (
    FlameRenderer,
    Struct,
    create_logger,
    get_data,
    get_rays,
    setup_random_seeds,
)
from utils.flame_dataset import FlameDataset, ValidationDataset
from utils.HeadNeRF.HeadNeRFNet import HeadNeRFNet
from utils.metrics import VGGPerceptualLoss, psnr


def initialize_parameters(config_path: str) -> Tuple[Struct, Dict]:
    """
    Initialize parameters for training using the given config path.

    Args:
        config_path (str): The path to the configuration file to load.

    Returns:
        Tuple[Struct, Dict]: A tuple containing the BaseOptions object and the
        parameters dictionary.
    """
    setup_random_seeds()
    with open(config_path, "r") as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
        parameters["lr"] = float(parameters["lr"])
    opt = Struct(**parameters["BaseOptions"])
    parameters["num_sample_coarse"] = opt.num_sample_coarse
    return opt, parameters


def initialize_model(opt: Struct, parameters: Dict, device: str) -> torch.nn.Module:
    """
    Initialize a HeadNeRFNet model for training.

    Args:
        opt (Struct): A BaseOptions object containing the prediction image size.
        parameters (Dict): A dictionary containing the "pretrain" and "path_to_weights"
            parameters.
        device (str): The device to move the model to.

    Returns:
        torch.nn.Module: The initialized HeadNeRFNet model.
    """
    model = HeadNeRFNet(opt, False, False)
    if parameters["pretrain"]:
        model.load_state_dict(torch.load(parameters["path_to_weights"]))
    model.to(device)
    model.train()
    return model


def initialize_optimizer(
    model: torch.nn.Module,
    parameters: Dict,
    offsets: Optional[Dict[str, torch.nn.Parameter]],
) -> optim.Optimizer:
    """
    Initialize an optimizer for the model with optional parameter groups for offsets.

    Args:
        model (torch.nn.Module): The model whose parameters need to be optimized.
        parameters (Dict): A dictionary containing training parameters, including 'lr' (learning rate)
                           and 'disentangled_loss' (a boolean indicating whether to use disentangled loss).
        offsets (Optional[Dict[str, torch.nn.Parameter]]): A dictionary of offset parameters to be added
                                                           to the optimizer, if 'disentangled_loss' is enabled.

    Returns:
        optim.Optimizer: An Adam optimizer initialized with the specified parameters and learning rates.
    """
    optimizer = optim.Adam(model.parameters(), lr=parameters["lr"], betas=(0.9, 0.999))
    if parameters["disentangled_loss"] and offsets:
        init_learn_rate = 0.01
        for offset_name, param in offsets.items():
            lr_multiplier = 1.5 if offset_name != "apea_offset" else 1.0
            optimizer.add_param_group(
                {"params": [param], "lr": init_learn_rate * lr_multiplier}
            )
    return optimizer


def setup_renderer(opt: Struct, parameters: Dict) -> FlameRenderer:
    """
    Set up the FlameRenderer for rendering head avatars.

    Args:
        opt (Struct): A BaseOptions object containing the prediction image size.
        parameters (Dict): A dictionary containing the "use_mouth" and "tex_mouth_path"
            parameters.

    Returns:
        FlameRenderer: The initialized FlameRenderer object.
    """
    return FlameRenderer(
        (opt.pred_img_size, opt.pred_img_size),
        use_mouth=parameters["use_mouth"],
        tex_mouth_path=parameters["tex_mouth_path"],
    )


def prepare_datasets(parameters: Dict, opt: Struct) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare datasets for training and validation.

    Args:
        parameters (Dict): A dictionary containing parameters such as path to training and
            validation data, image size, batch size, and mode.
        opt (Struct): A BaseOptions object containing the prediction image size.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation
            dataloaders.
    """
    train_dataset = FlameDataset(
        dataset_path_flame=parameters["path_to_train_npz"],
        dataset_path_img=parameters["path_to_train_img"],
        image_size=(opt.pred_img_size, opt.pred_img_size),
        transform=None,
        angle_limits_xyz=[[-30, 30], [130, 230], [-30, 30]],
        rand_z_limits=[0.7, 2.0],
        only_one_face=False,
        randomize_mode=True,
        noise_amplitude=0.0,
        params_amplitude=2.0,
        create_masks=True,
        create_predefined_angles=False,
        neck_max_bounds=[0.2, 0.2, 0.2],
        neck_min_bounds=[-0.2, -0.2, -0.2],
        jaw_max_bounds=[-0.1, 0.002, 0.002],
        jaw_min_bounds=[-1.25, -0.002, -0.002],
        mode=parameters["mode"],
        blur=parameters["blur"],
        epochs=parameters["epochs"],
        batch_size=parameters["batch_size"],
        batch_count=parameters["batch_count"],
        z_augm=parameters["z_augm"],
        synth_z_augm=parameters["synth_z_augm"],
        synth_augm=parameters["synth_augm"],
        synth_reg_strategy=parameters["synth_reg_strategy"],
        use_mouth=parameters["use_mouth"],
        tex_mouth_path=parameters["tex_mouth_path"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=parameters["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    validation_dataset = ValidationDataset(
        dataset_path_flame=parameters["path_to_val_npz"],
        dataset_path_img=parameters["path_to_val_img"],
        image_size=(opt.pred_img_size, opt.pred_img_size),
        transform=None,
        mode="beta" if parameters["mode"] in ["alpha", "beta"] else parameters["mode"],
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=parameters["batch_size"],
        shuffle=False,
        drop_last=True,
    )
    return train_dataloader, validation_dataloader


def main() -> None:
    opt, parameters = initialize_parameters("gamma.yaml")
    renderer = setup_renderer(opt, parameters)
    bg_value = 1.0 if opt.bg_type == "white" else 0.0

    writer, parameters, logdir_path = create_logger(parameters)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    offsets = None
    if parameters["disentangled_loss"]:
        offsets = {
            "shape_offset": torch.nn.Parameter(
                torch.zeros((1, 100), dtype=torch.float32).to(device),
                requires_grad=False,
            ),
            "pose_jaw_offset": torch.nn.Parameter(
                torch.zeros((1, 3), dtype=torch.float32).to(device), requires_grad=True
            ),
            "pose_neck_offset": torch.nn.Parameter(
                torch.zeros((1, 3), dtype=torch.float32).to(device), requires_grad=True
            ),
            "expr_offset": torch.nn.Parameter(
                torch.zeros((1, 100), dtype=torch.float32).to(device),
                requires_grad=True,
            ),
            "apea_offset": torch.nn.Parameter(
                torch.zeros((1, 127), dtype=torch.float32).to(device),
                requires_grad=False,
            ),
        }

    model = initialize_model(opt, parameters, device)
    optimizer = initialize_optimizer(model, parameters, offsets)

    train_dataloader, validation_dataloader = prepare_datasets(parameters, opt)

    ray_xy, ray_uv = get_rays(opt, parameters["batch_size"])
    ray_xy = ray_xy.to(device=device)
    ray_uv = ray_uv.to(device=device)

    criterion = torch.nn.MSELoss()
    vgg_loss_func = VGGPerceptualLoss(resize=True).to(device)

    for param in [
        *model.fg_CD_predictor.FeaExt_module_0.parameters(),
        *model.fg_CD_predictor.FeaExt_module_1.parameters(),
        *model.fg_CD_predictor.FeaExt_module_2.parameters(),
        *model.fg_CD_predictor.FeaExt_module_3.parameters(),
    ]:
        param.requires_grad = False

    for epoch in range(parameters["epochs"] + 1):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_masked_psnr = 0
        epoch_vgg_loss = 0
        epoch_bg_loss = 0
        epoch_mse = 0
        epoch_mask_mse = 0
        epoch_nonhead_loss = 0

        if parameters["disentangled_loss"]:
            epoch_shape_loss = 0
            epoch_expr_loss = 0
            epoch_pose_jaw_loss = 0
            epoch_pose_neck_loss = 0
            epoch_apea_loss = 0

        N = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, data in progress_bar:
            if parameters["disentangled_loss"]:
                shape_code, appea_code, Rmats, Tvecs, inv_inmat = get_data(
                    data,
                    opt,
                    {
                        "shape_offset": shape_offset,
                        "pose_jaw_offset": pose_jaw_offset,
                        "pose_neck_offset": pose_neck_offset,
                        "expr_offset": expr_offset,
                        "apea_offset": apea_offset,
                    },
                )
            else:
                shape_code, appea_code, Rmats, Tvecs, inv_inmat = get_data(data, opt)

            output = model(
                mode="train",
                batch_xy=ray_xy,
                batch_uv=ray_uv,
                bg_code=opt.bg_type,
                shape_code=shape_code,
                appea_code=appea_code,
                batch_Rmats=Rmats,
                batch_Tvecs=Tvecs.unsqueeze(-1),
                batch_inv_inmats=inv_inmat,
            )

            tv = output["coarse_dict"]["bg_img"] - bg_value
            bg_loss = torch.mean(tv * tv)

            mse_loss = criterion(
                output["coarse_dict"]["merge_img"],
                data["image"],
            )

            mask_batch = data["mask"].bool()
            mask_mse_loss = criterion(
                output["coarse_dict"]["merge_img"][mask_batch],
                data["image"][mask_batch],
            )

            masked_gt_img = data["image"].clone()
            masked_gt_img[~mask_batch] = bg_value

            vgg_loss = vgg_loss_func(output["coarse_dict"]["merge_img"], masked_gt_img)

            tv = output["coarse_dict"]["merge_img"][~mask_batch] - bg_value
            nonhead_loss = torch.mean(tv * tv)

            loss = (
                bg_loss * parameters["losses"]["bg_loss"]
                + mask_mse_loss * parameters["losses"]["mask_mse_loss"]
                + mse_loss * parameters["losses"]["mse_loss"]
                + nonhead_loss * parameters["losses"]["nonhead_loss"]
            )

            if parameters["disentangled_loss"]:
                shape_loss = torch.mean(shape_offset * shape_offset)
                expr_loss = torch.mean(expr_offset * expr_offset)
                pose_jaw_loss = torch.mean(pose_jaw_offset * pose_jaw_offset)
                pose_neck_loss = torch.mean(pose_neck_offset * pose_neck_offset)
                apea_loss = torch.mean(apea_offset * apea_offset)

                epoch_shape_loss += shape_loss
                epoch_expr_loss += expr_loss
                epoch_pose_jaw_loss += pose_jaw_loss
                epoch_pose_neck_loss += pose_neck_loss
                epoch_apea_loss += apea_loss

                loss += (
                    shape_loss * parameters["losses"]["shape_code"]
                    + expr_loss * parameters["losses"]["expr_code"]
                    + pose_jaw_loss * parameters["losses"]["pose_code"]
                    + pose_neck_loss * parameters["losses"]["pose_code"]
                    + apea_loss * parameters["losses"]["apea_code"]
                )

            loss += vgg_loss * parameters["losses"]["vgg_loss"]

            if torch.isnan(loss).any():
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu()
            epoch_psnr += (
                psnr(output["coarse_dict"]["merge_img"], data["image"]).detach().cpu()
            )
            epoch_masked_psnr += (
                psnr(
                    output["coarse_dict"]["merge_img"][data["mask"].bool()],
                    data["image"][data["mask"].bool()],
                )
                .detach()
                .cpu()
            )
            epoch_bg_loss += bg_loss.detach().cpu()
            epoch_mse += mse_loss.detach().cpu()
            epoch_mask_mse += mask_mse_loss.detach().cpu()
            epoch_vgg_loss += vgg_loss.detach().cpu()
            epoch_nonhead_loss += nonhead_loss.detach().cpu()

            progress_bar.set_description(
                f"Epoch {epoch}, ep_loss: {(epoch_loss / N):.4f}, Loss: {loss.item():.4f}"
            )

            N += 1
            if N > parameters["batch_count"]:
                break

        epoch_loss = epoch_loss.item() / N
        epoch_psnr = epoch_psnr.item() / N
        epoch_masked_psnr = epoch_masked_psnr.item() / N
        epoch_bg_loss = epoch_bg_loss.item() / N
        epoch_mse = epoch_mse.item() / N
        epoch_vgg_loss = epoch_vgg_loss.item() / N
        epoch_nonhead_loss = epoch_nonhead_loss.item() / N

        if parameters["disentangled_loss"]:
            epoch_shape_loss = epoch_shape_loss.item() / N
            epoch_expr_loss = epoch_expr_loss.item() / N
            epoch_pose_jaw_loss = epoch_pose_jaw_loss.item() / N
            epoch_pose_neck_loss = epoch_pose_neck_loss.item() / N
            epoch_apea_loss = epoch_apea_loss.item() / N

            writer.add_scalar("EpochShape/train", epoch_shape_loss, epoch)
            writer.add_scalar("EpochExpr/train", epoch_expr_loss, epoch)
            writer.add_scalar("EpochPoseJaw/train", epoch_pose_jaw_loss, epoch)
            writer.add_scalar("EpochPoseNeck/train", epoch_pose_neck_loss, epoch)
            writer.add_scalar("EpochApea/train", epoch_apea_loss, epoch)

        writer.add_scalar("EpochLoss/train", epoch_loss, epoch)
        writer.add_scalar("EpochPSNR/train", epoch_psnr, epoch)
        writer.add_scalar("EpochMaskesPSNR/train", epoch_masked_psnr, epoch)
        writer.add_scalar("EpochLossBG/train", epoch_bg_loss, epoch)
        writer.add_scalar("EpochMSE/train", epoch_mse, epoch)
        writer.add_scalar("EpochLossVGG/train", epoch_vgg_loss, epoch)
        writer.add_scalar("EpochNonHead/train", epoch_nonhead_loss, epoch)

        if epoch % 1 == 0:
            result = np.vstack(
                [
                    np.concatenate(
                        [
                            output["coarse_dict"]["merge_img"][j, ...]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy(),
                            data["image"][j, ...]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy(),
                        ],
                        axis=1,
                    )
                    for j in range(data["image"].shape[0])
                ]
            )

            img_batch = data["image"]
            mask_batch = data["mask"]

            train_img = torch.cat(
                [img_batch[k, ...] for k in range(img_batch.shape[0])], 2
            )
            train_mask = torch.cat(
                [mask_batch[k, ...] for k in range(img_batch.shape[0])], 2
            )
            train_masked = train_img * train_mask

            train_data = (
                torch.cat([train_img, train_mask, train_masked], 1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            writer.add_image(
                "Predicted/GT Images",
                torch.from_numpy(result).permute(2, 0, 1),
                epoch,
            )
            writer.add_image(
                "Train data (image, mask, masked image)",
                torch.from_numpy(train_data).permute(2, 0, 1),
                epoch,
            )

        if epoch % 100 == 0:
            result = np.vstack(
                [
                    np.concatenate(
                        [
                            output["coarse_dict"]["merge_img"][j, ...]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy(),
                            data["image"][j, ...]
                            .permute(1, 2, 0)
                            .detach()
                            .cpu()
                            .numpy(),
                        ],
                        axis=1,
                    )
                    for j in range(data["image"].shape[0])
                ]
            )
            img_batch = data["image"]
            mask_batch = data["mask"]

            train_img = torch.cat(
                [img_batch[k, ...] for k in range(img_batch.shape[0])], 2
            )
            train_mask = torch.cat(
                [mask_batch[k, ...] for k in range(img_batch.shape[0])], 2
            )
            train_masked = train_img * train_mask

            train_data = (
                torch.cat([train_img, train_mask, train_masked], 1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            writer.add_image(
                "Predicted100/GT Images",
                torch.from_numpy(result).permute(2, 0, 1),
                epoch,
            )
            writer.add_image(
                "Train data (image, mask, masked image)",
                torch.from_numpy(train_data).permute(2, 0, 1),
                epoch,
            )

            torch.save(
                model.state_dict(), os.path.join(logdir_path, f"model_{epoch}.pt")
            )

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), os.path.join(logdir_path, "last.pt"))
    writer.close()


if __name__ == "__main__":
    main()
