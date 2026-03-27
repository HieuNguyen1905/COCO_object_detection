import torch
import numpy as np
from tqdm import tqdm


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_ap(precisions, recalls, num_points=11):
    """
    Calculate Average Precision (AP) using n-point interpolation.

    Args:
        precisions: List of precision values
        recalls: List of recall values
        num_points: Number of interpolation points (default: 11)

    Returns:
        Average Precision score
    """
    precisions = np.array(precisions)
    recalls = np.array(recalls)

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]

    # n-point interpolation
    ap = 0
    for threshold in np.linspace(0, 1, num_points):
        if np.sum(recalls >= threshold) == 0:
            precision = 0
        else:
            precision = np.max(precisions[recalls >= threshold])
        ap += precision / num_points

    return ap


def calculate_map(predictions, targets, num_classes, iou_threshold=0.5, num_points=11):
    """
    Calculate mean Average Precision (mAP) for object detection.

    Args:
        predictions: List of dicts with 'boxes', 'labels', 'scores'
        targets: List of dicts with 'boxes', 'labels'
        num_classes: Number of classes (including background)
        iou_threshold: IoU threshold for matching predictions to ground truth
        num_points: Number of interpolation points for AP calculation

    Returns:
        mAP score
    """
    aps = []

    # Debug info
    total_predictions = sum(len(pred["boxes"]) for pred in predictions)
    total_targets = sum(len(target["boxes"]) for target in targets)

    # Check score distribution
    all_scores = []
    for pred in predictions:
        if len(pred["scores"]) > 0:
            all_scores.extend(pred["scores"].cpu().numpy().tolist())

    if len(all_scores) > 0:
        print(f"Debug: Total predictions: {total_predictions}, Total targets: {total_targets}")
        print(f"Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}], Mean: {np.mean(all_scores):.4f}")
    else:
        print(f"Debug: No predictions! Total targets: {total_targets}")

    for class_id in range(1, num_classes):  # Skip background class
        class_predictions = []
        class_targets = []

        # Collect predictions and targets for this class
        for pred, target in zip(predictions, targets):
            pred_boxes = pred["boxes"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()

            target_boxes = target["boxes"].cpu().numpy()
            target_labels = target["labels"].cpu().numpy()

            # Filter by class
            pred_mask = pred_labels == class_id
            target_mask = target_labels == class_id

            if pred_mask.sum() > 0:
                for box, score in zip(pred_boxes[pred_mask], pred_scores[pred_mask]):
                    class_predictions.append({"box": box, "score": score})

            if target_mask.sum() > 0:
                for box in target_boxes[target_mask]:
                    class_targets.append({"box": box, "matched": False})

        if len(class_targets) == 0:
            continue

        # Debug: Show class with predictions and targets
        if len(class_predictions) > 0 or len(class_targets) > 0:
            print(f"Class {class_id}: {len(class_predictions)} preds, {len(class_targets)} targets")

        # Sort predictions by score
        class_predictions = sorted(class_predictions, key=lambda x: x["score"], reverse=True)

        # Calculate precision and recall for each prediction
        precisions = []
        recalls = []
        true_positives = 0
        false_positives = 0

        for pred in class_predictions:
            # Find best matching target
            best_iou = 0
            best_target_idx = -1

            for idx, target in enumerate(class_targets):
                if target["matched"]:
                    continue

                iou = calculate_iou(pred["box"], target["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = idx

            # Check if prediction matches a target
            if best_iou >= iou_threshold:
                class_targets[best_target_idx]["matched"] = True
                true_positives += 1
            else:
                false_positives += 1

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / len(class_targets)

            precisions.append(precision)
            recalls.append(recall)

        # Calculate AP for this class
        if len(precisions) > 0:
            ap = calculate_ap(precisions, recalls, num_points=num_points)
            aps.append(ap)
            print(f"  -> AP: {ap:.4f}")

    final_map = np.mean(aps) if len(aps) > 0 else 0.0
    print(f"Final mAP from {len(aps)} classes: {final_map:.4f}")
    return final_map


@torch.no_grad()
def evaluate_map(model, dataloader, device, num_classes, iou_threshold=0.5, num_points=11):
    """
    Evaluate mAP on a dataset.

    Args:
        model: Detection model
        dataloader: DataLoader for evaluation
        device: Device to run on
        num_classes: Number of classes (including background)
        iou_threshold: IoU threshold for matching
        num_points: Number of interpolation points for AP calculation

    Returns:
        mAP score
    """
    model.eval()

    all_predictions = []
    all_targets = []

    print("Evaluating mAP...")
    for images, targets in tqdm(dataloader, desc="Inference"):
        images = [img.to(device) for img in images]
        predictions = model(images)

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    print("Calculating mAP metrics...")
    mAP = calculate_map(all_predictions, all_targets, num_classes, iou_threshold, num_points)
    return mAP
