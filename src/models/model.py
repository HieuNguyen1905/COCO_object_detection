from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int, pretrained: bool = True, base_model: str = "fasterrcnn_resnet50_fpn"):

    base_model = base_model.lower()
    weights = "DEFAULT" if pretrained else None
    if base_model == "fasterrcnn_resnet50_fpn":
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        raise ValueError(f"Unsupported base model: {base_model}")

    # Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model    