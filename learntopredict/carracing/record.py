# -*- coding: utf-8 -*-
import numpy as np
import os
import imageio

class ImageRecorder:
    def __init__(self):        
        self.save_dir = "results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.counter = 0

    def concat_images(self, images):
        """ Concatenate images into one image """
        image_height = images[0].shape[0]
        spacer = np.ones([image_height, 1, 3], dtype=np.uint8) * 255
        images_with_spacers = []
            
        image_size = len(images)
            
        for i in range(image_size):
            images_with_spacers.append(images[i])
            if i != image_size-1:
                # Add 1 pixel space
                images_with_spacers.append(spacer)
        ret = np.hstack(images_with_spacers)
        return ret

    def record(self, img_input, obs_real, obs_pred, peeked, vae):
        img_pred = vae.decode(obs_pred.reshape(1, -1)) * 255.0
        img_pred = np.round(img_pred).astype(np.uint8)
        img_pred = img_pred.reshape(64, 64, 3)
        
        img_dec = vae.decode(obs_real.reshape(1, -1)) * 255.0
        img_dec = np.round(img_dec).astype(np.uint8)
        img_dec = img_dec.reshape(64, 64, 3)

        img_input = np.round(img_input * 255.0).astype(np.uint8).reshape(64, 64, 3)

        img_all = self.concat_images([img_input, img_dec, img_pred])
        imageio.imwrite(os.path.join(self.save_dir,
                                     "" + (format(self.counter, "05d") + ".png")),
                        img_all)
        
        self.counter += 1
