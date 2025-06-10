#!/usr/bin/env python

import cv2
from sklearn.cluster import KMeans

img = cv2.imread('image.jpg')

img_flattened = img.reshape(-1, 3)

kmeans = KMeans(n_clusters=3)
kmeans.fit(img_flattened)
img_segmented = kmeans.cluster_centers_[kmeans.labels_].astype('uint8')
img_segmented = img_segmented.reshape(img.shape)
cv2.imwrite('image_kmeans.jpg', img_segmented)

# SAM 2 by Meta
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(img_rgb)
    masks, _, _ = predictor.predict()

img_sam2 = (masks.transpose(1, 2, 0)*255).astype('uint8')
cv2.imwrite('image_sam2.jpg', img_sam2)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(img_rgb)
    masks, _, _ = predictor.predict(point_coords=[[[5, 5], [30, 50]]],
                                    point_labels=[[0, 1]],
                                    multimask_output=True)

img_sam2 = (masks.transpose(1, 2, 0)*255).astype('uint8')
cv2.imwrite('image_sam2_person.jpg', img_sam2)
