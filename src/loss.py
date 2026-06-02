import torch
import torch.nn as nn

from utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self, S, B, C) -> None:
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)

        # We have set num_boxes_pred_per_cell to be 2 (B),
        # there for we need to take best box out of them
        # then calculate the loss on best box and forget others
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) # target will be same since that is original box to predict
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)

        is_box_exist = target[..., 20].unsqueeze(3) # this act as identity function, that only calculate loss on cell where box exist

        # ============================================== #
        # loss For Box Coordinates
        # ============================================== #
        box_predictions = is_box_exist * (
            (
                bestbox * predictions[..., 26:30] # since bestbox is an index telling which box is best out of 2 [0 th or 1 th], therefore bestbox = 1 ,means second box is the best, so we set second box here
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = is_box_exist * target[..., 21:25]