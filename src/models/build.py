from .detectors.faster_rcnn import build_faster_rcnn

def build_model(cfg):
    if cfg["name"] == "faster_rcnn":
        return build_faster_rcnn(cfg["num_classes"])