import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 640,
                         mean: list[float] = (0.485, 0.456, 0.406),
                         std: list[float] = (0.229, 0.224, 0.225)) -> A.Compose:

    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,  # Đặt để fill có hiệu lực
                fill=114, # Đặt giá trị pixel của padding 114 vì nằm giữa 0 và 255
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2), 
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", # xmin, ymin, xmax, ymax
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )


def get_val_transforms(image_size: int = 640,
                       mean: list[float] = (0.485, 0.456, 0.406),
                       std: list[float] = (0.229, 0.224, 0.225)) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=114,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )

def build_train_transforms(image_size: int, mean: list[float], std: list[float]) -> A.Compose:
    train_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,  # Đặt để fill có hiệu lực
                fill=114, # Đặt giá trị pixel của padding 114 vì nằm giữa 0 và 255
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2), 
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", # xmin, ymin, xmax, ymax
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )
    return train_transform

def build_val_transforms(image_size: int, mean: list[float], std: list[float]) -> A.Compose:
    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=114,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )
    return val_transform