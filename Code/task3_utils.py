from data_utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import torch

IMG_WIDTH = 1280
IMG_HEIGHT = 720

NET_SIZE = (112, 112)

MAX_VAL = 700 # Anything larger than this distance in the images, is discarded

# The capacity of the containers in mL
ANNOTATION = {1: 520.0, # Red cup
              2: 185.0, # Small white cup
              3: 202.0, # Small transparent cup
              4: 296.0, # Green glass
              5: 363.0, # Wine glass
              6: 128.0, # Champagne flute
              7: 3209.397, # Cereal box
              8: 1239.84, # Biscuits box
              9: 471.6} # Tea box

def computeObjectBoundingBox(depth_img_path, debug=False):
    ''' Given a depth image, try to find the region where the object is located

    '''

    # Define height limit to search, since bottom area is always a table
    HEIGHT_LIMIT = 600
    # Minimum area of the contours to be considered
    MIN_CONTOUR_AREA = 15000
    # In percentage (of bounding box)
    BBOX_EXPANSION = 1.05

    # Load image as uint16
    cv_img = cv2.imread(depth_img_path, -1)[:HEIGHT_LIMIT]

    if debug:
        # Show image
        plt.imshow(cv_img)
        plt.title("Original Depth Image")
        plt.show()

    # Find stats of image
    #mean = np.mean(cv_img)
    #std = np.std(cv_img)

    # Define distance threshold
    DIST_THRESHOLD = 700
    #DIST_THRESHOLD = mean - std

    # Filter pixels by distance threshold
    filter1 = np.where(cv_img < DIST_THRESHOLD, cv_img, 0)

    # Extract contours
    # Convert to unsigned 8-bit
    filter1_8u = ((filter1 / filter1.max()) * 255).astype(np.uint8)
    # Apply closing operation, try to retrieve some of the "missing" regions
    kernel = np.ones((15, 15), np.uint8)
    filter1_8u = cv2.morphologyEx(filter1_8u, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(filter1_8u, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        if debug:
            print("Couldn't find any contours")
            
        return cv_img, None

    if debug:
        viz_img = np.dstack([filter1_8u.copy(), filter1_8u.copy(), filter1_8u.copy()])

    # Iterate over contours, find the largest one
    #candidates = []
    bestContour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_CONTOUR_AREA and bestContour is None:
            bestContour = contour
        elif area > MIN_CONTOUR_AREA and bestContour is not None:
            # Use contour to find mask
            curr_cnt_mask = np.zeros(cv_img.shape, np.uint8)
            cv2.drawContours(curr_cnt_mask, [contour], 0, 255, -1)
            
            best_cnt_mask = np.zeros(cv_img.shape, np.uint8)
            cv2.drawContours(best_cnt_mask, [bestContour], 0, 255, -1)
            
            best_cnt_mean = cv2.mean(cv_img, mask=best_cnt_mask)
            curr_cnt_mean = cv2.mean(cv_img, mask=curr_cnt_mask)
            
            if curr_cnt_mean[0] < best_cnt_mean[0]:
                bestContour = contour
            
            if debug:
                plt.imshow(curr_cnt_mask, cmap='gray')
                plt.title("Current contour mask")
                plt.show()
                plt.imshow(best_cnt_mask, cmap='gray')
                plt.title("best contour mask")
                plt.show()            

    # Given the largest contour, find the bounding box
    if bestContour is not None:
        #print("Found object contour!")

        # If we found our object contour, find the bounding box
        x, y , w, h = cv2.boundingRect(bestContour)

    else:
        if debug:
            print("Couldn't determine object contour")
        return cv_img, None
    
    # Expand the bounding box a bit
    inc_w_h = int((w * BBOX_EXPANSION - w) / 2)
    inc_h_h = int((h * BBOX_EXPANSION - h) / 2)
    
    topx = x - inc_w_h
    topy = y - inc_h_h
    botx = x + w + inc_w_h
    boty = y + h + inc_h_h
    
    # Make sure values stay within range
    if topx < 0:
        topx = 0
    if topy < 0:
        topy = 0
    if botx > cv_img.shape[1]:
        botx = cv_img.shape[1]
    if boty > cv_img.shape[0]:
        boty = cv_img.shape[0]
    
    pt1 = (topx, topy)
    pt2 = (botx, boty)

    if debug:
        cv2.rectangle(viz_img, pt1, pt2, (255, 0, 0), 3)    

        cv2.drawContours(viz_img, contours, -1, (0, 255, 0), 2)

        plt.imshow(filter1)
        plt.title(f"Cropped to `DIST_THRESHOLD={DIST_THRESHOLD}` pixels")
        plt.show()

        plt.imshow(filter1_8u, cmap='gray')
        plt.title(f"Unsigned 8 image")
        plt.show()

        plt.imshow(viz_img, cmap='gray')
        plt.title(f"Viz image")
        plt.show()

    return cv_img, (filter1, pt1, pt2)
    
def findBestFrameROI(depth_imgs, search_max=30):
    _index = -1
    if len(depth_imgs) < search_max:
        search_n = len(depth_imgs) - 1
    else:
        search_n = search_max
        
    for _ in range(search_n):
        depth_img, ret = computeObjectBoundingBox(depth_imgs[_index], debug=False)
        
        if ret != None:
            filter1, (topx, topy), (botx, boty) = ret

            roi = filter1[topy:boty, topx:botx]
            
            return roi

        _index += -1
        
    return None

def load_depth_roi_dataset(dataset_root):
    ''' Iterate over all combinations of the dataset and try to retrieve the
        depth ROI that contains the object.

    '''

    data = []
    targets = []
    for obj_id in range(1, 10):
        print(f"Extracting data from object id: `{obj_id}`")
        for sit in s_dict.keys():
            for fi in fi_dict.keys():
                for fu in fu_dict.keys():
                    for b in b_dict.keys():
                        for l in l_dict.keys():
                            try:
                                sample = retrieve_data(dataset_root, obj_id, s=sit, fi=fi, fu=fu, b=b, l=l)
                            except Exception as e:
                                pass
                                #print(f"No sample for combination: {(obj_id, sit, fi, fu, b, l)}")
                            if sample != -1:
                                ret = findBestFrameROI(sample['depth'][2])
                                if ret is None:
                                    print(f"Failed to find good ROI for {sample['depth'][2][-1]}")
                                    #failed += 1
                                    #failed_samples.append((sit, fi, fu, b, l))
                                else:
                                    # Add sample to dataset
                                    roi_h, roi_w = ret.shape
                                    
                                    # Normalize ROI by max val
                                    data.append((np.divide(ret, MAX_VAL), roi_h/IMG_HEIGHT, roi_w/IMG_WIDTH))

                                    # Rescale the targets to be between 0.0~1.
                                    targets.append(ANNOTATION[obj_id]/4000)

    return data, targets

def construct_pytorch_dataset(data, targets, test_size, batch_size, plot_graphs=False):
    ''' Performs train/val split and return ready to train pytorch data loaders '''

    # Analyze class distribution
    _classes, counts = np.unique(targets, return_counts=True)
    if plot_graphs:
        print(_classes, counts)
        plt.bar(_classes, counts)
        plt.title("Whole dataset")
        plt.show()

    # Split in train and test
    X_train, X_test, y_train, y_test = train_test_split(data, targets,
                                                       test_size=0.15)

    if plot_graphs:
        _classes, counts = np.unique(y_train, return_counts=True)
        n_train_samples = len(y_train)
        print(_classes, counts)
        plt.bar(_classes, counts)
        plt.title("Train dataset")
        plt.show()

        _classes, counts = np.unique(y_test, return_counts=True)
        print(_classes, counts)
        plt.bar(_classes, counts)
        plt.title("Test dataset")
        plt.show()

    class VolumeDataset(Dataset):
        def __init__(self, x, y):
            self.images = []
            self.rois_size = []
            
            for _x in x:
                self.images.append(cv2.resize(_x[0], (112, 112)))
                self.rois_size.append((_x[1], _x[2]))
            
            self.y_volume = y
            
            assert len(self.images) == len(self.rois_size)
            assert len(self.images) == len(self.y_volume)
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return (torch.Tensor(self.images[idx]).unsqueeze(0), torch.Tensor(self.rois_size[idx])), torch.Tensor([self.y_volume[idx]])
        
    train_dataset = VolumeDataset(X_train, y_train)
    test_dataset = VolumeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                              num_workers=0)

    return train_loader, test_loader

def getModelPrediction(model, depth_frames):
    roi = findBestFrameROI(depth_frames)
    
    if roi is not None:
        h, w = roi.shape
        h = h / IMG_HEIGHT
        w = w / IMG_WIDTH
        
        # Normalize and resize roi
        roi = np.divide(roi, MAX_VAL)
        roi = cv2.resize(roi, NET_SIZE)
        
        # Convert everything to tensors
        tensor_roi = torch.Tensor(roi).unsqueeze(0)
        tensor_roi = tensor_roi.unsqueeze(0).cuda()
        tensor_roi_info = torch.Tensor([h, w]).unsqueeze(0).cuda()

        # Feed to model and get prediction
        with torch.no_grad():
            pred = model(tensor_roi, tensor_roi_info)
            return pred.item() * 4000
    else:
        print("Couldnt find roi")
        return None
