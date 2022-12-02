# -*- coding: utf-8 -*-
"""
Experiments for resilience of AttentionUnets with foreground ratios for Spleen MSD data.
"""

import sys
import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

# import matplotlib.pyplot as plt

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
)

from monai.networks.nets import AttentionUnet

# from monai.networks.layers import Norm
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config

import wandb


def check_loader(val_files, val_transforms):
    """
    CHECK_LOADER verifies that the data paths are setup correctly.
    """
    check_ds = Dataset(data=val_files, transform=val_transforms)
    data_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(data_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 64]
    # plt.figure("check", (12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("image")
    # plt.imshow(image[:, :, 64], cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("label")
    # plt.imshow(label[:, :, 64])
    # plt.show()


def generate_dataset(spleen_dir):
    """
    CREATE_DATASET creates a list of files for training and validation.
    """
    train_images = sorted(glob(os.path.join(spleen_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob(os.path.join(spleen_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files, val_files, test_files = (
        data_dicts[:26],
        data_dicts[26:31],
        data_dicts[31:],
    )
    return train_files, val_files, test_files


# Logging spleen slices to W&B
# utility function for generating interactive image mask from components
def wb_mask(bg_img, mask):
    """
    WB_MASK is a wandb helper to generate masks.
    """
    return wandb.Image(
        bg_img,
        masks={
            "ground truth": {
                "mask_data": mask,
                "class_labels": {0: "background", 1: "mask"},
            }
        },
    )


def log_slices(train_files, val_transforms, total_slices=100):
    """
    LOG_SLICES is a wandb helper to log images.
    """
    wandb_mask_logs = []
    wandb_img_logs = []

    check_ds = Dataset(data=train_files, transform=val_transforms)
    data_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(data_loader)  # get the first item of the dataloader

    image, label = (check_data["image"][0][0], check_data["label"][0][0])

    for img_slice_no in range(total_slices):
        img = image[:, :, img_slice_no]
        lbl = label[:, :, img_slice_no]

        # append the image to wandb_img_list to visualize
        # the slices interactively in W&B dashboard
        wandb_img_logs.append(wandb.Image(img, caption=f"Slice: {img_slice_no}"))

        # append the image and masks to wandb_mask_logs
        # to see the masks overlayed on the original image
        wandb_mask_logs.append(wb_mask(img, lbl))

    wandb.log({"Image": wandb_img_logs})
    wandb.log({"Segmentation mask": wandb_mask_logs})


def main(input_image_size, root_dir):
    """
    Runs the training and validation loop for Spleen data segmentation.
    """
    print_config()

    # Set MSD Spleen dataset
    data_dir = os.path.join(root_dir, "data", "raw")
    spleen_dir = os.path.join(data_dir, "Task09_Spleen")
    train_files, val_files, test_files = generate_dataset(spleen_dir)

    # Define Configuration
    config = {
        # data
        "cache_rate": 1.0,
        "num_workers": 2,
        "seed": np.random.randint(0, 100),
        "input_image_size": input_image_size,
        # train settings
        "train_batch_size": 2,
        "val_batch_size": 1,
        "learning_rate": 1e-4,
        "max_epochs": 500,
        "val_interval": 5,  # check validation score after n epochs
        # AttentionUnet model
        "model_type": "AttentionUnet",  # just to keep track
        "model_params": dict(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        ),
    }

    # 🐝 initialize a wandb run
    wandb_run = wandb.init(
        project="MONAI_Spleen_3D_Segmentation_AttentionUnet",
        config=config,
        dir=os.path.join(root_dir),
    )

    # Set deterministic training for reproducibility
    set_determinism(seed=config["seed"])

    # Setup transforms for training and validation
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(input_image_size, input_image_size, input_image_size),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
        ]
    )

    # Check DataLoader
    check_loader(val_files, val_transforms)

    # Define CacheDataset and DataLoader for training and validation
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=config["cache_rate"],
        num_workers=config["num_workers"],
    )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=config["cache_rate"],
        num_workers=config["num_workers"],
    )
    # val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=config["val_batch_size"], num_workers=config["num_workers"]
    )

    # Create Model, Loss, Optimizer and Scheduler

    # standard PyTorch program style: create AttentionUnet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUnet(**config["model_params"]).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=False, reduction="mean"
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"], eta_min=1e-9)

    # Execute a typical PyTorch training process

    # 🐝 log images to W&B
    log_slices(train_files, val_transforms, total_slices=100)

    # 🐝 log gradients of the model to wandb
    wandb.watch(model, log_freq=100)

    max_epochs = config["max_epochs"]
    val_interval = config["val_interval"]
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    fg_ratio = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )

            this_fg_ratio = []
            for idx in range(labels.shape[0]):
                ratio = np.sum(labels[idx, ...]) / np.prod(labels[idx, ...].shape)
                this_fg_ratio.append(ratio)
                fg_ratio.append(ratio)
            wandb.log({"train/fg_ratio": np.mean(this_fg_ratio).item()})

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}, "
                f"train_loss: {loss.item():.4f}"
            )

            # 🐝 log train_loss for each step to wandb
            wandb.log({"train/loss": loss.item()})

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # step scheduler after each epoch (cosine decay)
        scheduler.step()

        # 🐝 log train_loss averaged over epoch to wandb
        wandb.log({"train/loss_epoch": epoch_loss})

        # 🐝 log learning rate after each epoch to wandb
        wandb.log({"learning_rate": scheduler.get_lr()[0]})

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (input_image_size, input_image_size, input_image_size)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size, sw_batch_size, model
                    )
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # 🐝 aggregate the final mean dice result
                metric = dice_metric.aggregate().item()

                # 🐝 log validation dice score for each validation round
                wandb.log({"val/dice_metric": metric})

                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(wandb_run.dir, "best_metric_model.pth"),
                    )
                    print("saved new best metric model")
                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print(
        f"\ntrain completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}"
    )

    # Save foreground ratio list to file
    np.savetxt(
        os.path.join(wandb_run.dir, "foreground_ratios.csv"),
        fg_ratio,
        delimiter=", ",
        fmt="%f",
    )

    # 🐝 log best score and epoch number to wandb
    wandb.log({"best_dice_metric": best_metric, "best_metric_epoch": best_metric_epoch})

    # 🐝 Version your model
    best_model_path = os.path.join(wandb_run.dir, "best_metric_model.pth")
    model_artifact = wandb.Artifact(
        "AttentionUnet",
        type="model",
        description="AttentionUnet for 3D Segmentation of spleen",
        metadata=dict(config["model_params"]),
    )
    model_artifact.add_file(best_model_path)
    wandb.log_artifact(model_artifact)

    # Log predictions to W&B in form of table
    test_ds = CacheDataset(
        data=test_files,
        transform=val_transforms,
        cache_rate=config["cache_rate"],
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=config["val_batch_size"], num_workers=config["num_workers"]
    )

    # 🐝 create a wandb table to log input image, ground_truth masks and predictions
    columns = ["filename", "dsc", "hausdorff", "abs_diff", "fg_ratio"]
    table = wandb.Table(columns=columns)
    os.makedirs(os.path.join(wandb_run.dir, "test_pred"), exist_ok=True)

    model.load_state_dict(
        torch.load(os.path.join(wandb_run.dir, "best_metric_model.pth"))
    )
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            # get the filename of the current image
            fn = (
                test_data["image_meta_dict"]["filename_or_obj"][0]
                .split("/")[-1]
                .split(".")[0]
            )
            roi_size = (input_image_size, input_image_size, input_image_size)
            sw_batch_size = 4
            test_outputs = sliding_window_inference(
                test_data["image"].to(device), roi_size, sw_batch_size, model
            )
            test_labels = test_data["label"].to(device)

            test_outputs = [post_pred(idx) for idx in decollate_batch(test_outputs)]
            test_labels = [post_label(idx) for idx in decollate_batch(test_labels)]

            n_seg = nib.Nifti1Image(test_outputs[0][1, ...].cpu().numpy(), np.eye(4))
            nib.save(
                n_seg, os.path.join(wandb_run.dir, "test_pred", f"pred{i:d}.nii.gz")
            )

            dice_metric(y_pred=test_outputs, y=test_labels)
            dsc_metric = dice_metric.aggregate().item()
            dice_metric.reset()

            hausdorff_metric(y_pred=test_outputs, y=test_labels)
            hd_metric = hausdorff_metric.aggregate().item()
            hausdorff_metric.reset()

            ratio = np.sum(test_labels[0][1, ...].cpu()) / np.prod(
                test_labels[0][1, ...].cpu().shape
            )
            abs_diff = np.sum(
                np.abs(test_outputs[0][1, ...].cpu() - test_labels[0][1, ...].cpu())
            ) / np.sum(test_labels[0][1, ...].cpu() == 1)

            # 🐝 Add data to wandb table dynamically
            table.add_data(fn, dsc_metric, hd_metric, abs_diff, ratio)

    # log predictions table to wandb with `val_predictions` as key
    wandb.log({"test_predictions": table})

    # 🐝 Close your wandb run
    wandb.finish()


if __name__ == "__main__":
    input_image_size = int(sys.argv[1])
    root_dir = str(sys.argv[2])
    main(input_image_size, root_dir)
