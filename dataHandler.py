from torch.utils.data import Dataset
import os 
import pandas as pd 
import torch
from PIL import Image

def get_odometer_box(dataFrame, idx):
    boxes = []
    ymin = dataFrame["ymin"][idx]
    ymax = dataFrame["ymax"][idx]
    xmin = dataFrame["xmin"][idx]
    xmax = dataFrame["xmax"][idx]
    
    boxes.append([xmin, ymin, xmax, ymax])
            
    return boxes 
class ImageDataset(Dataset):
        """Image dataset."""
        def __init__(self, df, root_dir, transform):
            #csv_file = csv_path + '/dataFrameMerged.csv'
            #df = pd.read_csv(csv_file)
            isAutomatic = df['odometer_type']==0
            df = df[isAutomatic]
        
            df = df.reset_index() 
            self.labelDF = df
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.labelDF)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            boxList = get_odometer_box(self.labelDF,idx)
            img_name = os.path.join(self.root_dir,
                                    self.labelDF['image'][idx])
            image = Image.open(img_name)
            image = self.transform(image)
            #labels = self.labelDF['odometer_type'][idx] # If number of classes are 2 and we want to classify the odometer type. 
            odoType =  [self.labelDF['odometer_type'][idx]]
            odoType = torch.as_tensor(odoType, dtype=torch.int64,)
            # pascal_voc format [x_min, y_min, x_max, y_max]
            #boxes = [self.labelDF['xmin'][idx],self.labelDF['ymin'][idx]]
            boxes = torch.as_tensor(boxList, dtype=torch.int32,)
            #area = (self.labelDF['xmax'][idx] - self.labelDF['xmin'][idx]  ) * (self.labelDF['ymax'][idx]  - self.labelDF['ymin'][idx])
            #area = torch.as_tensor(area, dtype=torch.int64,)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            image_id = torch.tensor([idx])
            
            target = {}
            target['boxes']     = boxes
            target['labels']   = odoType 
            target['area']      = area
            target['iscrowd']   = iscrowd
            target["image_id"]  = image_id
            
            return image, target

