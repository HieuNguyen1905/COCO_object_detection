from torchvision.models.detection import fasterrcnn_resnet50_fpn

def build_faster_rcnn(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.roi_heads.box_predictor.cls_score.out_features = num_classes
    return model