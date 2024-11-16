import torch
import numpy as np


def get_batch_aabb_pair_ious(batch_boxes_1_bound, batch_boxes_2_bound):
    box_1_x_min, box_1_y_min, box_1_z_min = torch.tensor_split(batch_boxes_1_bound[:, 0], 3, dim=1)
    box_1_x_max, box_1_y_max, box_1_z_max = torch.tensor_split(batch_boxes_1_bound[:, 1], 3, dim=1)

    box_2_x_min, box_2_y_min, box_2_z_min = torch.tensor_split(batch_boxes_2_bound[:, 0], 3, dim=1)
    box_2_x_max, box_2_y_max, box_2_z_max = torch.tensor_split(batch_boxes_2_bound[:, 1], 3, dim=1)

    x_a = torch.maximum(box_1_x_min, box_2_x_min)
    y_a = torch.maximum(box_1_y_min, box_2_y_min)
    z_a = torch.maximum(box_1_z_min, box_2_z_min)
    x_b = torch.minimum(box_1_x_max, box_2_x_max)
    y_b = torch.minimum(box_1_y_max, box_2_y_max)
    z_b = torch.minimum(box_1_z_max, box_2_z_max)

    zero_tensor = torch.zeros_like(x_a)
    intersection_volume = torch.maximum((x_b - x_a), zero_tensor) * torch.maximum((y_b - y_a), zero_tensor) * \
                          torch.maximum((z_b - z_a), zero_tensor)
    box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
    box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)
    iou = intersection_volume / (box_1_volume + box_2_volume - intersection_volume + torch.finfo(torch.float32).eps)
    return iou.flatten()


def batched_nms_3d(boxes, scores, proposal_batch_id, iou_threshold):
    keep_indices = []
    for batch_id in proposal_batch_id.unique():
        batch_mask = proposal_batch_id == batch_id
        boxes_batch = boxes[batch_mask]
        scores_batch = scores[batch_mask]

        _, sorted_indices = scores_batch.sort(descending=True)
        selected_indices = []

        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            selected_indices.append(current_idx.item())
            if len(sorted_indices) == 1:
                break
            
            current_box = boxes_batch[current_idx].unsqueeze(0)
            rest_boxes = boxes_batch[sorted_indices[1:]]
            ious = iou_3d(current_box, rest_boxes).squeeze(0)

            low_iou_mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][low_iou_mask]

        selected_global_indices = torch.nonzero(batch_mask).flatten()[selected_indices]
        keep_indices.extend(selected_global_indices.tolist())

    return torch.tensor(keep_indices, device=boxes.device, dtype=torch.long)


def iou_3d(boxes1, boxes2):
    N = boxes1.size(0)
    M = boxes2.size(0)

    min_corner1 = boxes1[:, 0, :]
    max_corner1 = boxes1[:, 1, :]
    min_corner2 = boxes2[:, 0, :]
    max_corner2 = boxes2[:, 1, :]

    min_corner1 = min_corner1.unsqueeze(1).expand(N, M, 3)
    max_corner1 = max_corner1.unsqueeze(1).expand(N, M, 3)
    min_corner2 = min_corner2.unsqueeze(0).expand(N, M, 3)
    max_corner2 = max_corner2.unsqueeze(0).expand(N, M, 3)

    inter_min = torch.max(min_corner1, min_corner2)
    inter_max = torch.min(max_corner1, max_corner2)
    inter_dims = torch.clamp(inter_max - inter_min, min=0)
    inter_vol = inter_dims.prod(2)

    vol1 = (max_corner1 - min_corner1).prod(2)
    vol2 = (max_corner2 - min_corner2).prod(2)
    
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou