#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 3D MRI image
import nibabel as nib
nib_data = nib.load('Iguana.nii.gz')
mri_data = nib_data.get_fdata() 
height, width = mri_data.shape[:2]
new_width = 50
aspect_ratio = width / height
new_height = int(new_width / aspect_ratio)
mri_data_resized = cv2.resize(mri_data, (new_width, new_height))
mri_data_resized = mri_data_resized[:,:,::5]

fig, ax = plt.subplots(figsize=(1, 1), facecolor='white', frameon=False)
ax.set_xticks([])
ax.set_yticks([])
_ = [s.set_visible(False) for s in ax.spines.values()]
ax_list = []
for i in range(mri_data_resized.shape[2]):
    ax_list.append([ax.imshow(mri_data_resized[:,:,i], animated=True, cmap='gray')])
ani = animation.ArtistAnimation(fig, ax_list, interval=50, blit=True, repeat_delay=1000)
writer = animation.PillowWriter(fps=15, metadata=dict(artist='Miquel'), bitrate=1800)
ani.save('iguana_animation.gif', writer=writer)
