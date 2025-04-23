"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(weights=True) # pretrained is depricated

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        for level_name, feature_shape in dummy_out_shapes:
            B, C, H, W = feature_shape
            self.fpn_params[f'p{level_name[-1]}'] = torch.nn.Conv2d(C, self.out_channels, 1)
            self.fpn_params[f'head{level_name[-1]}'] = torch.nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        scales = self.fpn_strides
        
        # First compute lateral convolutions for all levels
        fpn_laterals = {}
        fpn_laterals['c3'] = self.fpn_params['p3'](backbone_feats['c3'])
        fpn_laterals['c4'] = self.fpn_params['p4'](backbone_feats['c4'])
        fpn_laterals['c5'] = self.fpn_params['p5'](backbone_feats['c5'])
        
        # Top-down pathway - merge features
        # Start with p5 (no upsampling needed for the top level)
        merged_features = {}
        merged_features['p5'] = fpn_laterals['c5']
        
        # p4 = lateral_conv(c4) + upsample(p5)
        merged_features['p4'] = fpn_laterals['c4'] + F.interpolate(
            merged_features['p5'], 
            size=fpn_laterals['c4'].shape[-2:], # this is safer choice than using scale
            mode='nearest'
        )
        
        # p3 = lateral_conv(c3) + upsample(p4)
        merged_features['p3'] = fpn_laterals['c3'] + F.interpolate(
            merged_features['p4'], 
            size=fpn_laterals['c3'].shape[-2:], 
            mode='nearest'
        )
        
        # Apply 3x3 convs to get the final outputs
        fpn_feats['p3'] = self.fpn_params['head3'](merged_features['p3'])
        fpn_feats['p4'] = self.fpn_params['head4'](merged_features['p4'])
        fpn_feats['p5'] = self.fpn_params['head5'](merged_features['p5'])
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        B, C, H, W = feat_shape

        # NOTE: Better version (claude's version) is commented out, but just for the reference.
        # grid_y, grid_x = torch.meshgrid(
        #     torch.arange(H, device=torch.device(device)),
        #     torch.arange(W, device=torch.device(device)),
        #     indexing='ij'
        # )
        
        # # Convert grid coordinates to absolute image coordinates
        # # Add 0.5 to get center of each cell
        # grid_y = (grid_y + 0.5) * level_stride
        # grid_x = (grid_x + 0.5) * level_stride
        
        # # Stack X and Y coordinates and reshape to (H*W, 2)
        # coordinates = torch.stack([grid_y, grid_x], dim=-1).to(dtype)
        # location_coords[level_name] = coordinates.reshape(-1, 2)

        grid = torch.zeros(H, W, 2, dtype=dtype, device=device)
        for h in range(H):
            for w in range(W):
                grid[h, w][0] = level_stride * (h + 0.5)
                grid[h, w][1] = level_stride * (w + 0.5)
        location_coords[level_name] = grid.reshape(-1, 2)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code

    '''
    NOTE: It is done by chatgpt
    '''
    # Sort by score (descending)
    
    assert boxes.ndim == 2 and boxes.size(1) == 4, "`boxes` must be [N,4]"
    assert scores.ndim == 1 and scores.size(0) == boxes.size(0)
    order = scores.argsort(descending=True)
    # Pre‑allocate output tensor
    keep = torch.empty_like(order)

    # Extract corners for easy slicing
    x1, y1, x2, y2 = boxes.T      # each is (N,)

    areas = (x2 - x1) * (y2 - y1)

    num_kept = 0
    while order.numel() > 0:
        i = order[0]
        keep[num_kept] = i
        num_kept += 1

        if order.numel() == 1:  # no more boxes to compare
            break

        # Select the remaining boxes
        order = order[1:]
        xx1 = torch.maximum(x1[i], x1[order])
        yy1 = torch.maximum(y1[i], y1[order])
        xx2 = torch.minimum(x2[i], x2[order])
        yy2 = torch.minimum(y2[i], y2[order])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        rem_areas = areas[order]
        union = (areas[i] + rem_areas - inter)

        iou = inter / union
        # Keep boxes with IoU <= threshold
        
        order = order[iou <= iou_threshold]

    return keep[:num_kept]


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
