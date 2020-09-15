import os

import cv2
import numpy as np
import logging

from internal.detectors import base_detection as od
from internal.detectors import detection_output as do
from . import darknet

log = logging.getLogger(__name__)

CUR_DIR = os.path.dirname(__file__)
TRACKED_OBJECTS = 1
DETECTION_THRESHOLD = 0.5

configPath = CUR_DIR + "/cfg/yolov4.cfg"
weightPath = CUR_DIR + "/weights/yolov4.weights"
metaPath = CUR_DIR + "/cfg/coco.data"
namePath = CUR_DIR + "/data/coco.names"


class Yolo_Dark_Detection(od.Base_Detection):
    '''
    Object detector based on YOLOv4 by Alexey Bochkovskiy
    https://github.com/AlexeyAB/darknet
    '''

    def __init__(self, avoider=None, class_zone_file=None, testmode=None, logfile=None):
        super(Yolo_Dark_Detection, self).__init__(avoider, testmode, logfile)
        self.height = 720
        self.width = 960
        self.xmargin = 70
        self.ymargin = 80

        if self.testmode == 'ratio_size':
            self.hull_threshold = 1.2
        else:
            self.hull_threshold = 1.15

        self.stop_threshold = 2.0
       
        self.net_main = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
        self.meta_main = darknet.load_meta(metaPath.encode("ascii"))

        self.network_width = darknet.network_width(self.net_main)
        self.network_height = darknet.network_height(self.net_main)

        # Darknet image used to hold image data during inference
        self.darknet_image = darknet.make_image(self.network_width, self.network_height , 3)

        # Output of the inference operation
        self.output = do.Detection_Output(self.network_width, self.network_height, TRACKED_OBJECTS)

        # Object class based zone weight values
        self.zone_weights_dict = {}

        self.read_zone_weight_file(class_zone_file)
        


    def process_frame(self, frame, old):
        """
        Base function for processing a frame to be displayed
        """
        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        output = image

        if self.enabled:
            output = self.show_inference(image)

        return output, output


    def show_inference(self, image):
        '''
        Detect object in a single image
        '''
        
        # Resize to fit in YOLOv4 network as configured in cfg/yolov4.cfg
        resized = cv2.resize(image, (self.network_width, self.network_height), interpolation=cv2.INTER_LINEAR)
    
        # Copy image data into a darknet image which can be used in the network
        darknet.copy_image_from_bytes(self.darknet_image, resized.tobytes())

        # Detect objects
        detections = darknet.detect_image(self.net_main, self.meta_main, self.darknet_image, thresh=DETECTION_THRESHOLD)
        
        # For all output bounding boxes, calculate some properties and add to output list if middle is in the 
        # Region of Interest
        for detection in detections:
            middle = detection[2][0], detection[2][1] # middle (x,y)
            if self.middle_in_roi(middle):
                coordinates = self.convert_back(detection[2][0], detection[2][1], detection[2][2], detection[2][3])
                size = detection[2][2] * detection[2][3] # width * height
                label = detection[0].decode()
                score = round(detection[1] * 100, 1)
                self.output.add_box(coordinates, size, middle, label, score)

        # Visualize ROI
        out = cv2.rectangle(resized, (self.xmargin, self.ymargin), (self.network_width - self.xmargin, 
                            self.network_height - self.ymargin), (0,0,255), 2)

        # If no objects are found, reset the output to clear the tracked object list
        if self.output.size() <= 0:
            self.output.reset()
            return cv2.resize(out, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Find tracked obstacles in the output, draw bounding boxes
        index_list = self.output.find_obstacles()
        out = self.cv_draw_boxes(out, index_list)
       

        # Get the size ratios of all obstacles in the obstacle tracking list
        size_ratio_list = self.output.get_size_ratio_list()
        print(size_ratio_list)
    
        # Go through the list and check if a size ratio exceeds the threshold
        # If one size ratio exceeds the stop threshold we quit looking and send stop command to the avoider
        avoid = False
        stop = False
        avoid_index = -1
        for i in range(len(size_ratio_list)):
            size_ratio = size_ratio_list[i]

            if size_ratio > self.stop_threshold and self.avoider is not None and self.avoider.enabled:
                self.avoider.avoid(True, None)
                avoid_index = i
                break

            if size_ratio >= self.hull_threshold:
                log.info("!!!! OBSTACLE DETECTED")
                log.info("Size ratio: " + str(size_ratio))
                avoid = True
                avoid_index = i
                break
            
        # If an obstacle is detected, show an image of the obstacle and calculate image zones
        # Then send command to the avoider
        if avoid:
            if self.testmode == "ratio_size":
                if self.avoider is not None and self.avoider.enabled:
                    with open(self.logfile, 'a') as f:
                        f.write(str(self.hull_threshold) + "," + str(size_ratio_list[avoid_index]) + ",")
                    cv2.imshow("Obstacle", out)
                    self.obstacle_detected = True
                    self.avoider.avoid(True, None, 'ratio')

            elif self.avoider is not None and self.avoider.enabled:
                if self.testmode == 'avoid':
                    with open(self.logfile, 'a+') as f:
                        f.write(str(size_ratio_list[avoid_index]) + ",")
                cv2.imshow("Obstacle", out)
                self.obstacle_detected = True
                self.avoider.avoid(False, self.calculate_zones(self.output.cur_obstacle_list, size_ratio_list))
            else:
                log.info("No avoider enabled!!")


        # Clear the output for use in the next iteration and return output image
        self.output.clear()
        return cv2.resize(out, (self.width, self.height), interpolation=cv2.INTER_LINEAR)


    def middle_in_roi(self, middle):
        '''
        Checks if the point (x,y) given lies within the ROI (middle) of the image
        '''
        return middle[0] > self.xmargin and middle[0] <= self.network_width - self.xmargin and \
               middle[1] > self.ymargin and middle[1] <= self.network_height - self.ymargin

    def cv_draw_boxes(self, img, index_list):
        '''
        Draw bounding boxes contained in 'detections' on the image
        Normal objects will be coloured green and obstacles will be coloured red
        '''

        for i in range(self.output.size()):
            # Calculate coordinates needed for opencv functions
            xmin, ymin, xmax, ymax = self.output.boxes[i].coordinates
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)

            display_str = self.output.boxes[i].label + ": " + str(self.output.boxes[i].score) + "%" 

            color = (0, 255, 0)

            # If current object is an obstacle, color red
            if i in index_list:
                color = (0,0,255)
                display_str += " TRACKING"

            # Write to frame
            cv2.rectangle(img, pt1, pt2, color, 1)
            cv2.putText(img, display_str, (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return img

    def convert_back(self, x, y, w, h):
        '''
        Convert bounding box values (x, y, width, height) to (xmin, ymin, xmax, ymax)
        Where (x,y) is the middle of the box
        '''

        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax


    def calculate_zones(self, obstacle_list, size_ratio_list):
        '''
        Calculate free space in the four image zones: Left Right Up Down by substracting intersecting space of obstacles
        '''

        zonel = zoner = (self.network_width / 2) * self.network_height
        zoneu = zoned = self.network_width * (self.network_height / 2)
        weights = [1,1,1,1]
        print("start loop")
        for i in range(len(obstacle_list)):
            # If obstacle in tracked list must be avoided (ratio > threshold)
            print("Obstacle" )
            print(obstacle_list[i])

            # Obstacle weight
            obstacle_weight = size_ratio_list[i] / self.hull_threshold
            
            box = obstacle_list[i].coordinates

            # Multiply weights of object class to total weights
            zone_weights = self.zone_weights_dict.get(obstacle_list[i].label)
            if zone_weights is not None:
                weights = [a*b for a,b in zip(weights, zone_weights)]
            else:
                log.warning(str(obstacle_list[i].label) + " is not in list")

            intersect = self.intersection(box, (0, 0, self.network_width/2-1, self.network_height-1))
            zonel -= intersect * obstacle_weight
            print("Zonel -= " + str(intersect))

            intersect = self.intersection(box, (self.network_width/2-1, 0, self.network_width-1, self.network_height-1))
            zoner -= intersect * obstacle_weight
            print("Zoner -= " + str(intersect))

            intersect = self.intersection(box, (0, 0, self.network_width-1, self.network_height/2-1))
            zoneu -= intersect * obstacle_weight
            print("Zoneu -= " + str(intersect))

            intersect = self.intersection(box, (0, self.network_height/2-1, self.network_width-1, self.network_height-1))
            zoned -= intersect * obstacle_weight
            print("Zoned -= " + str(intersect))
        
        # Create zones and multiply them with total zone weights derived from object classes of detected obstacles
        zones = od.Zones(zonel, zoner, zoneu, zoned)
        zones.times(weights)
        log.info(zones) 

        return zones


    def intersection(self, a, b):
        '''
        Intersection of a single Axis
        '''
        intersection = 0

        dx = min(a[2], b[2]) - max(a[0], b[0])
        dy = min(a[3], b[3]) - max(a[1], b[1])

        if dx >= 0 and dy >= 0:
            intersection = dx * dy
            
        return intersection


    def read_zone_weight_file(self, path):
        '''
        Reads '.zones' weight file
        Assumed format: class,zonel,zoner,zoneu,zoned
        '''
        log.info("Reading class weight file " + path)
        try:
            with open(path, 'r') as file:
                for line in file:
                    line = line.rstrip("\n")
                    split = line.split(',')
                    self.zone_weights_dict[str(split[0])] = [float(x) for x in split[1:]]
        except:
            pass

        log.info(self.zone_weights_dict)
        

    def add_ratio(self, amount):
        if self.testmode == 'ratio_size':
            self.hull_threshold += amount
            round(self.hull_threshold, 3)


if __name__ == '__main__':
    logging.basicConfig(format="%(name)s: %(levelname)s: %(message)s" ,level=logging.INFO)
    IMG_PATH = "img/";
    img1 = cv2.imread(IMG_PATH + "desk2.png")
    img2 = cv2.imread(IMG_PATH + "desk2.png")

    detector = Yolo_Dark_Detection(None, class_zone_file="./internal/detectors/yolodark/data/coco_adj.zones")

    output = detector.show_inference(img1)
    output = cv2.line(output, (0, 360), (960, 360), (225,0,0))
    output = cv2.line(output, (480, 0), (480, 720), (225,0,0))
    cv2.imshow('Output 1', output)
    detector.calculate_zones(detector.output.cur_obstacle_list, detector.output.get_size_ratio_list())

    #output = detector.show_inference(img2)
    #cv2.imshow('Output 2', output) 
    
    cv2.waitKey(0)
