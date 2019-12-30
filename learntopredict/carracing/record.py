# -*- coding: utf-8 -*-
import numpy as np
import os
import skimage.draw
import cv2
import shutil
import subprocess


class MovieWriter(object):
    """ Movie output utility class. """
    def __init__(self, file_name, frame_size, fps):
        """
        frame_size is (w, h)
        """
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vout = cv2.VideoWriter()
        success = self.vout.open(file_name, fourcc, fps, frame_size, True)
        if not success:
            print("Create movie failed: {0}".format(file_name))

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.vout.write(frame)

    def close(self):
        self.vout.release()
        self.vout = None


class AnimGIFWriter(object):
    """ Anim GIF output utility class. """
    def __init__(self, file_name, fps, tmp_dir_name="tmp_gif"):
        """
        frame_size is (w, h)
        """
        self.file_name = file_name
        self.fps = fps
        self.tmp_dir_name = tmp_dir_name
        self.image_index = 0

        if os.path.exists(self.tmp_dir_name):
            shutil.rmtree(self.tmp_dir_name)
        os.mkdir(self.tmp_dir_name)

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        file_path = os.path.join(self.tmp_dir_name, "frame{0:03d}.png".format(self.image_index))
        cv2.imwrite(file_path, frame)
        self.image_index += 1
        
    def close(self):
        delay = int(round(100 / self.fps))
        command = "convert -delay {} {}/*.png {}".format(delay,
                                                         self.tmp_dir_name,
                                                         self.file_name)
        subprocess.call(command.split())
        shutil.rmtree(self.tmp_dir_name)


class ImageRecorder:
    def __init__(self, record_type='mov'):
        self.save_dir = "results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.counter = 0

        if record_type == 'gif':
            self.writer = AnimGIFWriter(os.path.join(self.save_dir, "car_racing.gif"),
                                        fps=15)
        else:
            self.writer = MovieWriter(os.path.join(self.save_dir, "car_racing.mov"),
                                      (64*3+2, 64), fps=15)


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

    def draw_frame(self, image):
        width = image.shape[0]
        
        rr0, cc0 = skimage.draw.line(      0,        0, width-1,       0)
        rr1, cc1 = skimage.draw.line(      0,        0,       0, width-1)
        rr2, cc2 = skimage.draw.line(      0,  width-1, width-1, width-1)
        rr3, cc3 = skimage.draw.line(width-1,        0, width-1, width-1)
        
        image[rr0, cc0] = [255,0,0]
        image[rr1, cc1] = [255,0,0]
        image[rr2, cc2] = [255,0,0]
        image[rr3, cc3] = [255,0,0]

    def record(self, img_input, obs_real, obs_pred, peeked, vae):
        img_pred = vae.decode(obs_pred.reshape(1, -1)) * 255.0
        img_pred = np.round(img_pred).astype(np.uint8)
        img_pred = img_pred.reshape(64, 64, 3)
        
        if peeked:
            self.draw_frame(img_pred)
        
        img_dec = vae.decode(obs_real.reshape(1, -1)) * 255.0
        img_dec = np.round(img_dec).astype(np.uint8)
        img_dec = img_dec.reshape(64, 64, 3)

        img_input = np.round(img_input * 255.0).astype(np.uint8).reshape(64, 64, 3)

        img_all = self.concat_images([img_input, img_dec, img_pred])
        
        """
        imageio.imwrite(os.path.join(self.save_dir,
                                     "" + (format(self.counter, "05d") + ".png")),
                        img_all)
        """

        self.writer.add_frame(img_all)
        
        self.counter += 1

    def close(self):
        self.writer.close()
    

    
