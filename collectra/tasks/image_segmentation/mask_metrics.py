"""Shared, class-aware foreground overlap for bitmap segmentation trainers."""


def mask_overlap(prediction, target, threshold):
    """Class-aware foreground intersection/union, independent of instance order.

    Validation targets use the resized image grid; RF-DETR's postprocessor uses
    original image dimensions. Compare unions on the target grid, excluding
    low-confidence predictions and retaining false positives in absent classes.
    """
    import torch
    from torch.nn.functional import interpolate

    masks = prediction["masks"]
    if masks.ndim == 4:
        masks = masks[:, 0]
    keep = prediction["scores"] >= threshold
    masks, labels = masks[keep], prediction["labels"][keep]
    truth = target["masks"].bool()
    if len(masks) and masks.shape[-2:] != truth.shape[-2:]:
        masks = interpolate(masks[:, None].float(), size=truth.shape[-2:], mode="nearest")[:, 0].bool()
    intersection = torch.zeros((), dtype=torch.int64, device=truth.device)
    union = intersection.clone()
    classes = torch.cat((labels, target["labels"])).unique()
    for label in classes:
        predicted = masks[labels == label].any(0) if (labels == label).any() else torch.zeros(
            truth.shape[-2:], dtype=torch.bool, device=truth.device,
        )
        actual = truth[target["labels"] == label].any(0)
        intersection += (predicted & actual).sum()
        union += (predicted | actual).sum()
    return torch.stack((intersection, union))


