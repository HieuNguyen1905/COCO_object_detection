from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int, pretrained: bool = True):
    """Build Faster R-CNN model with ResNet50 backbone.

    Args:
        num_classes: Number of classes (including background)
        pretrained: Use pretrained weights

    Returns:
        Faster R-CNN model
    """
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classifier head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model    