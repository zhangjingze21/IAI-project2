import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import models
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import Any, Dict, List, Tuple
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rich import print
import random
import glob

from models.classifer import ClassiferLitModule

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    model = ClassiferLitModule.load_from_checkpoint(cfg.ckpt_path)
    target_layers = [model.net.features[-4]]
    # todo: modify the code to enable more detailed result

    image_dir = "/home/stu5/scratch/dev/learning/IAI/pro2/project2/data/test"
    image_paths = glob.glob(f"{image_dir}/*/*.jpg")
    print(len(image_paths))
    # random sample one from the list
    image_path = random.choice(image_paths)
    image_idx = os.path.basename(image_path).split(".")[0]
    output_dir = f"outputs/{image_idx}"
    os.makedirs(output_dir, exist_ok=True)

    # save original image
    cv2.imwrite(os.path.join(output_dir, "original.jpg"), cv2.imread(image_path))

    rgb_img = cv2.imread(image_path)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(
        rgb_img,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ).to("cuda")

    for i in range(6):
        targets = [ClassifierOutputTarget(i)]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            cam.batch_size = 32
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                aug_smooth=False,
                eigen_smooth=False
            )
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        gb_model = GuidedBackpropReLUModel(model=model, device="cuda")
        gb = gb_model(input_tensor, target_category=None)
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        cam_gb = deprocess_image(cam_mask * gb)
        gb = deprocess_image(gb)

        cam_output_path = os.path.join(output_dir, f'gradcam_{i}.jpg')
        cv2.imwrite(cam_output_path, cam_image)

    # concat images, Landscape orientation
    model = model.to("cuda")
    model.eval()
    preditions = model(input_tensor)
    preditions = torch.softmax(preditions, dim=1)
    print(preditions)
    pred_idx = torch.argmax(preditions).item()

    images = []
    # white space
    images.append(np.ones((256, 256, 3), np.uint8) * 255)
    images.append(cv2.imread(image_path))
    for i in range(6):
        images.append(cv2.imread(os.path.join(output_dir, f'gradcam_{i}.jpg')))
    
    images = [cv2.resize(img, (256, 256)) for img in images]
    concat_img = np.concatenate(images, axis=1)
    # add pred label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(concat_img, f"Predicted: {pred_idx}", (10, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    # add gt label
    gt_label = image_path.split("/")[-2].split("_")[0]
    cv2.putText(concat_img, f"GT: {gt_label}", (10, 60), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # add prob of each class
    for i in range(6):
        cv2.putText(concat_img, f"{i}: {preditions[0, i]:.2f}", (10, 90 + 30 * i), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(os.path.join(output_dir, "concat.jpg"), concat_img)







if __name__ == '__main__':
    main()