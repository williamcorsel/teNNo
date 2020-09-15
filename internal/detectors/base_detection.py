import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

class Zones:
    """
    Class containing sizes of the 4 image zones surrounding an obstacle
    """
    
    def __init__(self, left, right, up, down):
        self.l = left
        self.r = right
        self.u = up
        self.d = down


    def __str__(self):
        return "Zones - l: " + str(self.l) + " r: " + str(self.r) + " u: " + str(self.u) + " d: " + str(self.d)


    def times(self, other):
        self.l *= other[0]
        self.r *= other[1]
        self.u *= other[2]
        self.d *= other[3]


class Base_Detection(object):
    """
    Baseclass for Obstacle detection
    """

    def __init__(self, avoider, testmode, logfile, log_level = logging.DEBUG):
        self.enabled = True
        log.setLevel(log_level)
        # Dimensions of frame to be processed and displayed
        self.height = 480
        self.width = 640
        self.xmargin = 70
        self.ymargin = 80

        # Obstacle avoider
        self.avoider = avoider

        self.min_score_thresh = 0.5
        self.obstacle_detected = False

        self.testmode = testmode
        self.logfile = logfile


    def enable(self):
        self.enabled = True


    def disable(self):
        self.enabled = False


    def toggle(self):
        self.enabled = not self.enabled


    def process_frame(self, frame, old):
        """
        Base function for processing a frame to be displayed
        """

        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)

        return image, image


    def create_mask(self):
        """
        Creates a mask to exclude the margins 
        """

        mask = np.zeros((self.height, self.width), np.uint8)
        cv2.rectangle(mask, (self.xmargin, self.ymargin), (self.width - self.xmargin, self.height - self.ymargin), 255, thickness=-1)
        return mask


    def prepare_workspace(self, frame):
        """
        Rescaled the raw image if needed
        """

        if frame.shape[0] != self.height or frame.shape[1] != self.width:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        return frame


    def separate_border(self, image):
        mask = self.create_mask()

        crop = image[self.ymargin:-self.ymargin+1, self.xmargin:-self.xmargin+1].copy()
        
        mask = cv2.bitwise_not(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        border = cv2.bitwise_and(image.copy(), mask)
        return crop, border


    def set_avoider(self, avoider):
        self.avoider = avoider


    def write_to_log(self, text):
        with open(self.logfile, 'a+') as f:
            f.write(str(text))
