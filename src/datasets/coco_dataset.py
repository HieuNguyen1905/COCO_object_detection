import os
import torch
import cv2
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODataset(Dataset):

    def __init__(self, img_dir, ann_file, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.transforms = transforms
        # Load COCO API
        self.coco = COCO(ann_file)
        
        self.ids = list(self.coco.imgs.keys()) # list of image ids
        # original_id -> 0-based index (for model)
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: idx for idx, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0] # dict with keys: 'file_name', 'width', 'height', ...  
        file_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id) # get annotation ids for the image
        anns = self.coco.loadAnns(ann_ids) # list of dicts with keys: 'bbox', 'category_id', 'area', 'iscrowd', ...
        bboxes = []
        labels = []

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.cat_id_to_label:
                continue  # skip categories not in our mapping (shouldn't happen if COCO is consistent)
            label = self.cat_id_to_label[cat_id] # convert original category id to 0-based label index
            bbox = ann["bbox"] # [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            bboxes.append([x_min, y_min, x_max, y_max])
            labels.append(label)

        # Apply albumentations transforms
        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=bboxes, labels=labels)
            img = transformed["image"]
            bboxes = transformed["bboxes"]
            labels = transformed["labels"]
        else:
            # If no transforms, convert to tensors
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        target = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        return img, target
    