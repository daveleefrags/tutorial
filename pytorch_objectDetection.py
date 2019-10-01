

### Copy of pytorch tutorial
### https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


import os
import numpy as np
import torch 
from PIL import Image  


## Make dataset class 
class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root 
        self.transforms = transforms  
        
        # Load all image files, sorting them to 
        # ensure that they are aligned
        # 이미지와 mask 파일의 리스트를 불러옵니다.
        self.imgs = list(sorted(os.path.join(root, "PNGImages")))
        self.masks = list(sorted(os.path.join(root, "PedMasks")))
        
    def __getitem__(self, idx):

        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        # Note that we haven't converted the mask to RGB, 
        # because each color corresponds to  different instance
        # with 0 being background
        # mask는 별도로 RGB로 컨버팅하지 않습니다. (3개의 채널로 이루어져 있지 않음)
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(img_path)
        mask = np.array(mask)  

        # mask에 있는 유니크한 값들을 가지고 옴
        # 첫번째 값은 0 - 즉 백그라운드이기 때문에 - 버리고 두번째부터 가지고 옴
        obj_ids = np.unique(mask) 
        obj_ids = obj_ids[1:]

        # Split the color-encoded coodinates for each mask of binary masks
        # 각각의 숫자로 되어있는 마스크를 여러장의 binary 마스크로 바꿔줍니다.
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids) 
        boxes = [] 
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax]) 

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64) 
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:, 1]) * (boxes[:,2] - boxes[:,0]) 
        
        iscrowded = torch.zeros((num_objs,), dtype=torch.int64)

        target = {} 
        target['boxes'] = boxes 
        target['labels'] = labels  
        target['masks'] = masks   
        target['image_id'] = image_id  
        target['area'] = area 
        target['iscrowd'] = iscrowd  

        if self.transforms is not None:
            img, target = self.transforms(img, target) 

        return img, target  

    def __len__(self):
        return len(self.imgs) 






# Option 1 :finetunning from a pretrained model
# pretrained된 모델을 불러와서 tunning 하는 방식입니다.

import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2 # 1 class (person) + background (사람이 있는 부분은 1, 배경은 0)
in_features = model.roi_heads.box_predictor.cls_score.in_features 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 





# Option 2 : Modifying the model to add a different backbone

import torchvision 
from torchvision.models.detection import FasterRCNN 
from torchvision.models.detection.rpn import AnchorGenerator 

# Load a pre-trained model for classification and return only the features 
# imagenet데이터로 pretrained된 mobilenet_v2모형을 로딩합니다.
# 이 모형은 feature를 가공하는 부분만 있으며, 이 뒤에 classifier부분을 추가합니다.
backbone = torchvision.models.mobilenet_v2(pretrained=True).features 

# FasterRCNN needs to know the number of
# output channels in a backbone. For Mobilenet_v2, It's 1280.
# so we need to add it here. 
# 이 백본 모형의 최종 아웃풋 사이즈에 대한 정보를 지정해줍니다.
backbone.out_channels = 1280

# Let's make the RPN generate 5X3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple because each feature 
# map could potentially have different sizes and aspect ratios.

anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256, 512),), 
                                   aspect_ratios = ((0.5, 1.0, 2.0),)
                                  )

# Let's define what are the feature maps that will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# If your backbone returns a Tensor, feature_names is expected to
# bo [0]. More generally, the backbone should return an OrderedDict[Tensor], 
# and in featuremap_names you can choose which feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(feature_names=[0],
                                               output_size=7,
                                               sampling_ratio=2  
                                              )

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)



# In our case, we want to fine-tune from a pre-trained model,
# given that our dataset is very small, so we will be following
# approach number 1 
# 이 경우, 데이터셋이 매우 작기 때문에 pretrained 모형을 활용하는
# 1번 방법으로 접근하도록 합니다.

# Here we want to also compute the instance segmentation masks,
# so we will be using Mask R-CNN 
# instance segmentation mask를 활용, Mask R-CNN을 활용하도록 합니다.

import torchvision 
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor 

def get_model_instance_segmentation(num_classes):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels 
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

