import time
import os
import logging

import torch
from torch.backends import cudnn
from matplotlib import colors
import cv2
import numpy as np

from .backbone import EfficientDetBackbone
from .efficientdet.utils import BBoxTransform, ClipBoxes
from .utils.utils import aspectaware_resize_padding, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from internal.detectors import detection_output as do
from internal.detectors import base_detection as od

log = logging.getLogger(__name__)

compound_coef = 0
force_input_size = None  # set None to use default size

anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size




model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + f'/weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()
model = model.cuda()

class Efficientdet_Detection(od.Base_Detection):
    def __init__(self, avoider=None):
        super(Efficientdet_Detection, self).__init__(avoider)
        self.height = 720
        self.width = 960
        self.xmargin = 60
        self.ymargin = 50
        self.size_threshold = 1.3
        self.output = do.Detection_Output((self.width - self.xmargin*2), (self.height - self.ymargin*2))


    def process_frame(self, frame, old):
        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        #image = self.prepare_workspace(image)
        crop, border = self.separate_border(image)

        
        output_frame = self.show_inference(crop)
       
        border[self.ymargin:-self.ymargin+1, self.xmargin:-self.xmargin+1] = output_frame
        cv2.rectangle(border, (self.xmargin, self.ymargin), (border.shape[1] - self.xmargin, border.shape[0] - self.ymargin), (0,0,255), 2)
        self.output.clear()
        return border, border


    def show_inference(self, image):
        _, framed_imgs, framed_metas = self.preprocess(image, max_size=input_size)

        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        #x = torch.from_numpy(framed_imgs).cuda()
        x = x.to(torch.float32).permute(0, 3, 1, 2)
     
        
        
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

       
        out = invert_affine(framed_metas, out)
        out = out[0]

        for i in range(len(out['rois'])):
            score = out['scores'][i]

            if score >= self.min_score_thresh:
                coordinates = out['rois'][i].astype(np.int)
                size = self.box_size(coordinates)
                middle = self.box_middle(coordinates)
                label = out['class_ids'][i]
                self.output.add_box(coordinates, size, middle, label, score)

        if self.output.size() <= 0:
            self.output.reset()
            return image

        index = self.output.find_obstacle()

        self.visualize_boxes(image, index)

        size_ratio = self.output.get_size_ratio(index)
        
        if size_ratio > self.size_threshold:
            log.info("!!!! OBSTACLE DETECTED")
            log.info("Size ratio: " + str(size_ratio))
            if self.avoider is not None and self.avoider.enabled:
                self.obstacle_detected = True
                cv2.imshow("Obstacle", image);
                stop = size_ratio > 2
                self.avoider.avoid(self.calculate_zones(self.output.get_coordinates(index), stop))

        return image


    def visualize_boxes(self, image, index):
        for i in range(self.output.size()):
            class_name = obj_list[self.output.get_label(i)]
            color = (0,255,0)
            if i == index:
                color = (0,0,255)
                class_name += '   TRACKING'
            
            plot_one_box(image, self.output.get_coordinates(i), class_name, self.output.get_score(i), color)


    def preprocess(self, ori_image, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        normalized_img = (ori_image / 255 - mean) / std
        img_meta = aspectaware_resize_padding(normalized_img, max_size, max_size,
                                                means=None)
        framed_img = img_meta[0] 
        framed_meta = img_meta[1:]
        
        return [ori_image], [framed_img], [framed_meta]


    def box_size(self, coords):
        """
        Returns normalized coordinates of bounding box
        Input: [ymin, xmin, ymax, xmax] (from output_dict['detection_boxes'])
        """
        return (coords[2] - coords[0]) * (coords[3] - coords[1])

    def box_middle(self, coords):
        """
        Returns coordinate of middle point of bounding box
        Input: (left, right, top, bottom)
        """
        box_middle = (coords[0] + (coords[2] - coords[0])/2, coords[1] + (coords[3] - coords[1])/2)
        return box_middle



    

if __name__ == '__main__':
    logging.basicConfig(format="%(name)s: %(levelname)s: %(message)s" ,level=logging.INFO)
    IMG_PATH = "img/";
    img1 = cv2.imread(IMG_PATH + "desk.png")
    img1 = cv2.resize(img1, (960, 720), interpolation=cv2.INTER_AREA)

    detector = Efficientdet_Detection()

    crop, border = detector.separate_border(img1)

    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    output = detector.show_inference(crop)

    border[detector.ymargin:-detector.ymargin+1, detector.xmargin:-detector.xmargin+1] = output
    border = cv2.rectangle(border, (detector.xmargin, detector.ymargin), (border.shape[1] - detector.xmargin, border.shape[0] - detector.ymargin), (0,0,255), 2)
    cv2.imshow("Out", border)
    
    cv2.waitKey(0)