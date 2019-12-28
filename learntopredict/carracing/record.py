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

    def record(self, img_input, obs_real, obs_pred, peeked, vae):
        img_pred = vae.decode(obs_pred.reshape(1, -1)) * 255.0
        img_pred = np.round(img_pred).astype(np.uint8)
        img_pred = img_pred.reshape(64, 64, 3)
        imageio.imwrite(os.path.join(self.save_dir,
                                     "pred_" + (format(self.counter, "05d") + ".png")),
                        img_pred)

        img_dec = vae.decode(obs_real.reshape(1, -1)) * 255.0
        img_dec = np.round(img_dec).astype(np.uint8)
        img_dec = img_dec.reshape(64, 64, 3)
        imageio.imwrite(os.path.join(self.save_dir,
                                     "dec_" + (format(self.counter, "05d") + ".png")),
                        img_dec)

        img_input = np.round(img_input * 255.0).astype(np.uint8).reshape(64, 64, 3)
        imageio.imwrite(os.path.join(self.save_dir,
                                     "input_" + (format(self.counter, "05d") + ".png")),
                        img_input)
        self.counter += 1
