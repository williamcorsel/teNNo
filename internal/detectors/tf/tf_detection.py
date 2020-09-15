import logging
import pathlib

import cv2
import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from internal.detectors import base_detection as od
from internal.detectors import detection_output as do


log = logging.getLogger(__name__)

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


'''
 Name of model used. More pretrained models can be found here:
 https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
'''
#MODEL_NAME = "ssd_mobilenet_v2_oid_v4_2018_12_12"
#MODEL_NAME = "rfcn_resnet101_coco_2018_01_28"
#MODEL_NAME = "faster_rcnn_inception_v2_coco_2018_01_28"
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'

'''
 List of the strings that is used to add correct label for each box.
 Make sure to match this file to the dataset the model is trained on
'''
#PATH_TO_LABELS = 'tenno/internal/detectors/models/labels/oid_v4_label_map.pbtxt'
PATH_TO_LABELS = 'internal/detectors/tf/models/labels/mscoco_label_map.pbtxt'



class Tf_Detection(od.Base_Detection):

    def __init__(self, avoider=None):
        super(Tf_Detection, self).__init__(avoider)
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
        self.model = self.load_model()

        self.size_threshold = 1.12
        self.output = do.Detection_Output((self.width - self.xmargin*2), (self.height - self.ymargin*2))

    def load_model(self):
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = MODEL_NAME + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=MODEL_NAME, 
            origin=base_url + model_file,
            untar=True)

        model_dir = pathlib.Path(model_dir)/"saved_model"
        log.info("Model saved to: " + str(model_dir))

        model = tf.saved_model.load(str(model_dir))
        
        model = model.signatures['serving_default']

        return model

    def process_frame(self, frame, old):
        """
        Base function for processing a frame to be displayed
        """

        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        image = self.prepare_workspace(image)
        crop, border = self.separate_border(image)

        
        output_frame = self.show_inference(crop);
        border[self.ymargin:-self.ymargin+1, self.xmargin:-self.xmargin+1] = output_frame

        # Draw rectangle to represent margin
        cv2.rectangle(border, (self.xmargin, self.ymargin), (border.shape[1] - self.xmargin, border.shape[0] - self.ymargin), (0,0,255), 2)
        self.output.clear()
        return border, border
       

    def show_inference(self, image):
    
        # Actual detection.
        output_dict = self.run_inference_for_single_image(image)
        # Visualization of the results of a detection.
        
       

        #boxes = self.get_boxes(output_dict)

        for i in range(len(output_dict['detection_boxes'])):
            score = output_dict['detection_scores'][i]

            if score >= self.min_score_thresh:
                coordinates = self.box_coordinates(output_dict['detection_boxes'][i])
                size = self.box_size(coordinates)
                middle = self.box_middle(coordinates)
                label = output_dict['detection_classes'][i]
                self.output.add_box(coordinates, size, middle, label, score)
        

        if self.output.size() <= 0:
            self.output.reset()
            return image

        index = self.output.find_obstacle()

        self.visualize_boxes(image, index)

        size_ratio = self.output.get_size_ratio(index)

       
        if size_ratio >= self.size_threshold:
            log.info("!!!! OBSTACLE DETECTED")
            log.info("Size ratio: " + str(size_ratio))
            if self.avoider is not None and self.avoider.enabled:
                self.obstacle_detected = True
                #cv2.imwrite("Obstacle", image);
                stop = size_ratio > 2
                self.avoider.avoid(self.calculate_zones(self.output.get_coordinates(index)), stop)

        return image


    def run_inference_for_single_image(self, image):
        
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        output_dict = self.model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
    
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

       
        
        # Handle models with masks:
        if 'detection_masks' in output_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
            
        return output_dict


    def visualize_boxes(self, image, index):
        for i in range(self.output.size()):
            xmin, xmax, ymin, ymax = self.output.get_coordinates(i)
        
            if self.output.get_label(i) in self.category_index.keys():
                class_name = self.category_index[self.output.get_label(i)]['name']
            else:
                class_name = 'N/A'

            display_str = '{}: {}%'.format(class_name, int(100*self.output.get_score(i)))
            color = (0,255,0)
            if i == index:
                color = (0,0,255)
                display_str += '   TRACKING'

            vis_util.draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax
                                                , color, 2, [display_str], False)


    def calculate_zones(self, box):
        zonel = box[0]
        zoner = (self.width - self.xmargin*2) - box[1]
        zoneu = box[2]
        zoned = (self.height - self.ymargin*2) - box[3]
        return od.Zones(zonel, zoner, zoneu, zoned)


    def box_coordinates(self, box):
        """
        Returns normalized coordinates of bounding box
        Input: [ymin, xmin, ymax, xmax] (from output_dict['detection_boxes'])
        """
        return (box[1] * (self.width - self.xmargin*2), box[3] * (self.width - self.xmargin*2),
                box[0] * (self.height - self.ymargin*2), box[2] * (self.height - self.ymargin*2))


    def box_size(self, box_coordinates):
        """
        Returns size of bounding box
        Input: (left, right, top, bottom)
        """
        return ((box_coordinates[1] - box_coordinates[0]) *
               (box_coordinates[3] - box_coordinates[2]))


    def box_middle(self, box):
        """
        Returns coordinate of middle point of bounding box
        Input: (left, right, top, bottom)
        """
        box_middle = (box[1] + (box[0] - box[1])/2, box[2] + (box[3] - box[2])/2)
        return box_middle

    def distance(self, x, y):
        return sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))
        



if __name__ == '__main__':
    logging.basicConfig(format="%(name)s: %(levelname)s: %(message)s" ,level=logging.INFO)
    IMG_PATH = "img/";
    img1 = cv2.imread(IMG_PATH + "desk.png")
    img1 = cv2.resize(img1, (640, 480), interpolation=cv2.INTER_AREA)
   

    detector = Tf_Detection(None)

    crop, border = detector.separate_border(img1)

    output = detector.show_inference(crop)

    border[detector.ymargin:-detector.ymargin+1, detector.xmargin:-detector.xmargin+1] = output

    # Draw rectangle to represent margin
    border = cv2.rectangle(border, (detector.xmargin, detector.ymargin), (border.shape[1] - detector.xmargin, border.shape[0] - detector.ymargin), (0,0,255), 2)
    border = cv2.resize(border, (960, 720), interpolation=cv2.INTER_AREA)


    cv2.imshow('Output', border)
    
    cv2.waitKey(0)