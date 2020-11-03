# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import imgproc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def groupbox(boxes):
    groupboxes = []
    i = 0
    start = 0
    group = []
    while(True):
        if abs(boxes[i][0][1] - boxes[start][0][1]) < 6:
            group.append(start+i)
            i += 1
        else:
            start = i
            groupboxes.append(group)
            group = []
        
        if i == len(boxes) - 1:
            break

    return groupboxes





def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)

        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        groupboxes = groupbox(boxes)
        top_boxes = np.array([boxes[i][0][1] for i in range(len(boxes))]).reshape(-1,1)
        kmeans = KMeans(n_clusters=11, random_state=0).fit(top_boxes)

        label  = kmeans.predict(top_boxes)
        # with open(res_file, 'w') as f:
        #     mask = np.zeros_like(img)
        #     for i, box in enumerate(boxes):
        #         # poly = np.array(box).astype(np.int32).reshape((-1))
        #         # strResult = ','.join([str(p) for p in poly]) + '\r\n'
        #         # f.write(strResult)

                
        #         # poly = poly.reshape(-1, 2)
        #         # cv2.polylines(mask, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)


        #         ptColor = (0, 255, 255)
        #         if verticals is not None:
        #             if verticals[i]:
        #                 ptColor = (255, 0, 0)

        #         if texts is not None:
        #             font = cv2.FONT_HERSHEY_SIMPLEX
        #             font_scale = 0.5
        #             cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
        #             cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)
        for i in np.unique(label):
            box = boxes[np.where(label == i)[0]]
            box = np.concatenate([np.expand_dims(box[i], axis = 0) for i in range(len(box))])
            y_min, y_max = np.min(box[:,:,1]), np.max(box[:,:,1])
            x_min, x_max = np.min(box[:,:,0]), np.max(box[:,:,0])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0,255,0),2)

        # Save result image
        cv2.imwrite(res_img_file, img)

