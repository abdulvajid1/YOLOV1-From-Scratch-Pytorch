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

        # ============================================== 
        # loss For Box Coordinates Loss
        # ============================================== 
        box_predictions = is_box_exist * (
            (
                bestbox * predictions[..., 26:30] # since bestbox is an index telling which box is best out of 2 [0 th or 1 th], therefore bestbox = 1 ,means second box is the best, so we set second box here
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = is_box_exist * target[..., 21:25]

        # some times the output will give negative value (for w, x due to training dynamics especially at starting or due to some explodes) 
        # which can cause taking sqrt on them (negative values) give nan value.
        # to avoid that, we do a simple trick (take sqrt on positve value then restore it's original sign)
        # torch.sign return 1 or -1 according to the sign of the value
        # we took the real sign of value 
        # then take abs of same value to make all of them positive before sqrt (sqrt on negative will nan)
        # multiply with real sign we took using torch sign to restore the sign of the value
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # =================================================
        # For Object loss
        # =================================================

        pred_box = (bestbox * predictions[..., 25: 26] # the number that tells if there is a box or not in the cell
                    + (1 - bestbox) * predictions[..., 20: 21])
        
        object_loss = self.mse(
            torch.flatten(is_box_exist * pred_box), 
            torch.flatten(is_box_exist * target[..., 20:21])
        )


        # =================================================
        # For No Object loss
        # =================================================
        
        # We do same as object loss but do 1 - exist_box, so this section only work when existbox = 0
        no_object_loss = self.mse(
            (1 - is_box_exist) * predictions[..., 20:21],
            (1 - is_box_exist) * target[..., 20:21]
        )

        # you may wonder why we have two loss here and not for object loss
        # in object loss, we only calculate for loss best box (higher iou box)
        # but for no object loss, we need to penalize for every box that wrongly tells there is box
        no_object_loss += self.mse(
            (1 - is_box_exist) * predictions[..., 25:26],
            (1 - is_box_exist) * target[..., 20:21]
        )

        # =================================================
        # For Class Loss
        # =================================================

        class_loss = self.mse(
            torch.flatten(is_box_exist * predictions[..., :20], end_dim=-2),
            torch.flatten(is_box_exist * target[... :20], end_dim=-2)
        )

        # =================================================
        # Full loss
        # =================================================
        loss = (
            self.lambda_coord * box_loss # first two row of loss in paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )