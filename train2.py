import torch
#import torch.optim as optim
import pandas as pd
#import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
torch.manual_seed(3407)
from torchvision import  transforms
#import matplotlib.pyplot as plt
# File import 
from dataHandler import ImageDataset
#from IoU import bb_intersection_over_union
#from model import Custom_predictor, CustomHead
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from torchvision.models.detection import FasterRCNN
#from torchvision.models.detection.rpn import AnchorGenerator
#import torchvision.models.detection._utils as det_utils

from tools.engine import train_one_epoch, evaluate
import tools.utils as utils 
import warnings
warnings.filterwarnings('ignore')
import time
start = time.time()

batchSize= 32
num_workers = 0
num_epochs =  25
num_classes = 4
learning_rate = 0.01
momentum = 0.92
csv_file = 'resized_DF.csv'
df = pd.read_csv(csv_file)
train_set,val_set=train_test_split(df,test_size=0.25,random_state=3407)
root_dir='../transferlearning_BB/images_224'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
                     transforms.ToTensor(),
                            #transforms.RandomHorizontalFlip(), 
                        transforms.Normalize((0.5,), (0.5,))
                            #transforms.Normalize(mean=[-1.5367e-04,  3.0208e-06,  3.2043e-04], std=[0.9998, 0.9999, 1.0002])
                            ])
train_dataset=ImageDataset( df=train_set,
                                #root_dir=args.data_path,
                                root_dir=root_dir,
                                transform = transform)

val_dataset=ImageDataset(   df=val_set,
                                #root_dir=args.data_path,
                                root_dir=root_dir,
                                transform = transform)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers= num_workers,
        collate_fn=utils.collate_fn
    )
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers= num_workers,
        collate_fn=utils.collate_fn
    )


def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model
model = get_object_detection_model(num_classes)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, val_loader, device=device)
end = time.time()
print('time', start-end)