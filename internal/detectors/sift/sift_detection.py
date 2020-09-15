import logging

import cv2
import numpy as np

from internal.detectors import base_detection as od

log = logging.getLogger(__name__)


"""
Obstacle detection algorithm using Object Size Expansion
Based on the paper: Obstacle Detection and Avoidance System Based on
Monocular Camera and Size Expansion Algorithm for UAVs by Abdulla Al-Kaff et al.
"""
class Sift_Detection(od.Base_Detection):
    def __init__(self, avoider=None, testmode=None, logfile=None):
        super(Sift_Detection, self).__init__(avoider, testmode, logfile)
        
        self.mask = self.create_mask()

        # Modules of the obstacle detector
        self.bf = cv2.BFMatcher()
        self.sift = cv2.xfeatures2d.SIFT_create()
    
        # Threshold values for obstacle
        if self.testmode == 'ratio_size':
            self.hull_threshold = 1.2
        else:
            self.hull_threshold = 1.151

        if self.testmode == 'ratio_kp':
            self.kp_threshold = 1.0
        else:
            self.kp_threshold = 1.09

        self.stop_hull_threshold = 1.6
        self.stop_kp_threshold = 1.4

        self.old_kp = None
        self.old_des = None

        # Debugging
        self.show_matches = False
        self.show_keypoints = False
        self.max_hull_ratio = 0.0
        self.max_kp_ratio = 0.0
        self.max_current = None
        self.max_previous = None

        
    def process_frame(self, frame, old):
        """
        Returns the correct frame to the video player:
        - Proccessed frame if detector is enabled
        - Raw frame if not
        """

        image = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_RGB2BGR)
        output = image
        cur_frame = None

        if self.enabled: # if obstacle avoidance is enabled
            cur_frame = self.prepare_workspace(image)

            _, cur_frame_processed = self.obstacle_detection(cur_frame, old)

            output = cur_frame_processed
        
        return cur_frame, output


    def obstacle_detection(self, cur_frame, old_frame):
        """
        Main obstacle detection function:
        - Keypoint detecting using SIFT for both frames
        - Keypoint matching using a Brute-force matcher
        - Convex hull calculations based on matched keypoints
        - Calls obstacle avoider if thresholds are exceeded
        """

        if old_frame is None or cur_frame is None:
            self.old_kp, self.old_des =  self.detect_keypoints(cur_frame);
            return old_frame, cur_frame

        frame = np.copy(cur_frame)
        hull_ratio, kp_ratio = self.calculate_ratio(frame, old_frame)

        if hull_ratio <= 0 or kp_ratio <= 0:
            return old_frame, cur_frame

        stop = False
        
        if self.testmode == 'ratio_size' and hull_ratio >= self.hull_threshold:
            if self.avoider is not None and self.avoider.enabled:
                self.disable()
                with open(self.logfile, 'a+') as f:
                    f.write(str(self.hull_threshold) + "," + str(hull_ratio) + ",")
                self.obstacle_detected = True
                cv2.imshow("Obstacle", frame)
                self.avoider.avoid(stop, None, testmode='ratio')

        elif self.testmode == 'ratio_kp' and kp_ratio >= self.kp_threshold:
            if self.avoider is not None and self.avoider.enabled:
                self.disable()
                with open(self.logfile, 'a+') as f:
                    f.write(str(self.kp_threshold) + "," + str(kp_ratio) + ",")
                self.obstacle_detected = True
                cv2.imshow("Obstacle", frame)
                self.avoider.avoid(stop, None, testmode='ratio')


        # If calculated values exceed threshold, enable the avoider
        elif kp_ratio >= self.kp_threshold and hull_ratio >= self.hull_threshold:
            log.info("!!!! OBSTACLE DETECTED")
            log.info("Hull ratio: " + str(hull_ratio) + " KP ratio: " + str(kp_ratio))
            if self.avoider is not None and self.avoider.enabled:
                self.obstacle_detected = True
                
                if self.testmode == 'avoid':
                    cv2.imshow("Obstacle", frame)
                    with open(self.logfile, 'a+') as f:
                        f.write(str(kp_ratio) + "," + str(hull_ratio) + ",")

                stop = hull_ratio > self.stop_hull_threshold and kp_ratio > self.stop_kp_threshold
                self.avoider.avoid(stop, self.calculate_zones(self.old_kp))
            
        
        return old_frame, frame
    

    def calculate_ratio(self, cur_frame, old_frame):

        # Detecting keypoints
        #old_kp, old_des = self.detect_keypoints(old_frame)
        cur_kp, cur_des = self.detect_keypoints(cur_frame)

        # Match and filter keypoints
        best = self.bf_matcher(self.old_kp, self.old_des, cur_kp, cur_des)

        # Draw rectangle to represent margin
        cur_frame = cv2.rectangle(cur_frame, (self.xmargin, self.ymargin), (cur_frame.shape[1] - self.xmargin, cur_frame.shape[0] - self.ymargin), (0,0,255), 2)

        # DEBUG: shows user the matches of the matcher
        if self.show_matches:
            gray1 = cv2.cvtColor(np.copy(old_frame), cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(np.copy(cur_frame), cv2.COLOR_BGR2GRAY)
            img = cv2.drawMatchesKnn(gray1, self.old_kp, gray2, cur_kp, best, None, singlePointColor=(0,0,255))
            cv2.imshow("Matches", img)
        
        # If no hull is found stop here
        if len(best) <= 3:
            self.old_kp = cur_kp
            self.old_des = cur_des
            return -1, -1
        
        # Calculate hull of previous frame
        old_hull = self.calculate_hull(self.kp_to_points(best, self.old_kp, False))
        old_hull_size = self.hull_size(old_hull)
        self.draw_hull(old_frame, old_hull)
       
        # Calculate hull of current frame
        cur_hull = self.calculate_hull(self.kp_to_points(best, cur_kp))
        cur_hull_size = self.hull_size(cur_hull)
        self.draw_hull(cur_frame, cur_hull)

        #Calculate hull ratio
        if cur_hull_size <= 0.0:
            hull_ratio = 0.0
        else:
            hull_ratio = cur_hull_size / old_hull_size;

        # Calculate keypoint ratio
        total = 0
        best_cur_kp = []
        for m in best:
            prev_size = self.old_kp[m[0].queryIdx].size
            cur_size = cur_kp[m[0].trainIdx].size
            best_cur_kp.append(cur_kp[m[0].trainIdx])
            total += cur_size / prev_size

        kp_ratio = total / len(best)
        
        #DEBUG: save max kp ratio and hull ratio to show at shutdown
        if kp_ratio > self.max_kp_ratio and hull_ratio > self.max_hull_ratio:
            self.max_kp_ratio = kp_ratio
            self.max_hull_ratio = hull_ratio
            self.max_previous = old_frame
            self.max_current = cur_frame

        self.old_kp = cur_kp
        self.old_des = cur_des

        if self.show_matches:
            log.debug("Hull ratio: " + str(hull_ratio) + " KP ratio: " + str(kp_ratio))

        return hull_ratio, kp_ratio


    def detect_keypoints(self, frame):
        """
        Detects keypoints in a frame using the SIFT algorithm included in OpenCV
        The mask is used to exclude the borders from being scanned
        Returns an array with found keypoints and an array with corresponding keypoint descriptors
        """
        
        colour = np.copy(frame)

        # Detect and compute keypoints on the grayscale image
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, self.mask)

        #DEBUG show found keypoints to the user
        if self.show_keypoints:
            cv2.imshow("Sift keypoints", cv2.drawKeypoints(gray, kp, colour, (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    
        return kp, des


    def bf_matcher(self, old_kp, old_des, cur_kp, cur_des):
        """
        Matches keypoints in 2 frames using the Brute Force matcher included in OpenCV
        Filters keypoints based on values described in the Size Expansion paper
        """

        # Error checking
        if old_des is None or cur_des is None or len(old_des) <= 3 or len(cur_des) <= 3:
            return []

        # Matching
        matches = self.bf.knnMatch(old_des, cur_des, 2)

        # Filtering
        good = []
        for m,n in matches:
            if m.distance <= 0.28 * n.distance: # check if distance ratio is lower than 0.28 -> value stated in paper
                good.append([m])

        best = []
        for m in good:
            prev_size = old_kp[m[0].queryIdx].size
            cur_size = cur_kp[m[0].trainIdx].size
       
            if cur_size > prev_size: # check if keypoint size in current frame is bigger than in the old frame
                best.append(m)

        return best


    def kp_to_points(self, best, cur_kp, train=True):
        """
        Converts keypoint to points on the frame (x,y)-coordinates
        - best should contain output of the BFMatcher
        - cur_kp should contain list of keypoints of one frame
        - train decides if cur_kp is list from train or test frame (old- vs cur- frame here)
        """

        # Create index list of keypoints
        index_list = []
        for m in best:
            if(train):
                index = m[0].trainIdx
            else:
                index = m[0].queryIdx
            index_list.append(index)

        # Convert to points
        points = cv2.KeyPoint_convert(cur_kp, index_list)
        return points


    def calculate_hull(self, points):
        """
        Creates a convex hull based on an array of points. This hull will include all points in the array
        """

        points = np.asarray(points)
        hull = cv2.convexHull(points)
        hull = np.array(hull).reshape((-1,1,2)).astype(np.int32)
        return hull


    def hull_size(self, hull):
        """
        Size of a hull created by cv2.convexHull
        """

        return cv2.contourArea(hull)


    def draw_hull(self, frame, hull, colour=(0, 255, 0)):
        """
        Draws a hull created by cv2.convexHull
        """

        cv2.drawContours(frame, [hull], -1, colour)


    def set_debug_mode(self, to):
        self.show_matches = to
        self.show_keypoints = to


    def calculate_zones(self, kp):
        """
        Calculates image zones based on the outer points
        """

        pl, pr, pu, pd = self.get_outer_points(kp)

        zonel = pl[0]
        zoner = ((self.width - 2*self.xmargin) - pr[0])
        zoneu = pu[1]
        zoned = ((self.height - 2*self.ymargin) - pd[1])

       
        zone_str = "zonel: " + str(zonel) + " zoner: " + str(zoner) + " zoneu: " + str(zoneu) + " zoned: " + str(zoned)
        log.debug("Zones ## " + zone_str)

        return od.Zones(zonel, zoner, zoneu, zoned)

    
    def get_outer_points(self, kp):
        """
        Finds the 4 most outer points of the object hull
        """
        assert self.xmargin is not None and self.ymargin is not None

        xmin = ymin = 100000
        xmax = ymax = -1

        for point in kp:
            pt = point.pt
            x = pt[0] - self.xmargin
            y = pt[1] - self.ymargin
            pt = (x, y)

            if pt[0] < xmin:
                pl = pt
                xmin = pt[0]
            if pt[1] < ymin:
                pu = pt
                ymin = pt[1]
            if pt[0] > xmax:
                pr = pt
                xmax = pt[0]
            if pt[1] > ymax:
                pd = pt
                ymax = pt[1]
        
        
        points_str = "pl: (" + str(pl[0]) + "," + str(pl[1]) + ") "
        points_str += "pr: (" + str(pr[0]) + "," + str(pr[1]) + ") "
        points_str += "pd: (" + str(pd[0]) + "," + str(pd[1]) + ") "
        points_str += "pu: (" + str(pu[0]) + "," + str(pu[1]) + ") "
        log.debug("Obstacle points ## " + points_str)

        return pl, pr, pu, pd


    def add_ratio(self, amount):
        if self.testmode == 'ratio_size':
            self.hull_threshold += amount
            round(self.hull_threshold, 3)
        elif self.testmode == 'ratio_kp':
            self.kp_threshold += amount
            round(self.kp_threshold, 3)


if __name__ == '__main__':
    logging.basicConfig(format="%(name)s: %(levelname)s: %(message)s" ,level=logging.DEBUG)
    IMG_PATH = "img/";
    img1 = cv2.imread(IMG_PATH + "4.png")
    img2 = cv2.imread(IMG_PATH + "5.png")
    img1 = cv2.resize(img1, (640, 480), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (640, 480), interpolation=cv2.INTER_AREA)


    detector = Sift_Detection(None)
    detector.set_debug_mode(True)
    detector.old_kp, detector.old_des = detector.detect_keypoints(img2)
    old, new = detector.obstacle_detection(img1, img2)

    stacked = np.concatenate((old, new), 1)
    blended = cv2.addWeighted(old, 0.5, new, 0.5, 0.0)

    cv2.imshow('Stacked', stacked)
    cv2.imshow("Blended", blended)
    
    cv2.waitKey(0)