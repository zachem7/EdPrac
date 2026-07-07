from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


def resize_image_and_boxes(image: Image.Image, boxes: torch.Tensor, image_size: int):
    width, height = image.size
    scale_x = image_size / width
    scale_y = image_size / height
    image = image.resize((image_size, image_size))
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return image, boxes


class GTSDBDetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int,
        train: bool = True,
        label_offset: int = 1,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(self.manifest_path)
        self.images = sorted(self.df["image_path"].unique().tolist())
        self.groups = {image: group for image, group in self.df.groupby("image_path")}
        self.image_size = image_size
        self.label_offset = label_offset

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        group = self.groups[image_path]

        boxes = group[["x1", "y1", "x2", "y2"]].to_numpy(dtype="float32")
        labels = group["class_id"].to_numpy(dtype="int64") + self.label_offset

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        image, boxes_tensor = resize_image_and_boxes(image, boxes_tensor, self.image_size)
        image_tensor = F.to_tensor(image)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
            "area": (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1]),
            "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64),
            "orig_size": torch.tensor([height, width]),
        }

        return image_tensor, target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
