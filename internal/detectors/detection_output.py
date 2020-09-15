import logging
import copy

from collections import defaultdict

from internal.util import distance


log = logging.getLogger(__name__)

class Detection_Output:
    '''
    Class for storing output of YOLOv4 object detector
    Tracks a list of obstacles found in subsequent frames
    '''

    def __init__(self, useable_width, useable_height, obstacle_list_size=1):
        self.boxes = [] # All found objects in frame
        self.obstacle_list_size = obstacle_list_size
        self.cur_obstacle_list = [] # All objects that are deemed obstacles and are tracked
        self.useable_width = useable_width
        self.useable_height = useable_height


    def add_box(self, coordinates, size, middle, label, score):
        '''
        Add new object to object detected list
        '''
        box = Box_Properties(coordinates, size, middle, distance(middle, (self.useable_width/2, self.useable_height/2))
                            , label, score)
        self.boxes.append(box)
        

    def size(self):
        return len(self.boxes)


    def find_obstacles(self):
        '''
        Compares obstacle tracking list with object detected list to update object sizes or track new objects
        '''
        index_list = []
        remove_set = set()

        # If frame has detected same object
        for i in range(len(self.cur_obstacle_list)):
            try:
                # Check if obstacle in list is still found in current frame
                index = self.boxes.index(self.cur_obstacle_list[i])
                index_list.append(index)
                old_size = self.cur_obstacle_list[i].old_size
                self.cur_obstacle_list[i] = copy.deepcopy(self.boxes[index])
                self.cur_obstacle_list[i].old_size = old_size

            except ValueError as err:
                # Object not found -> add to list to delete
                remove_set.add(i)
             
        # Remove obstacles that can't be tracked anymore
        self.cur_obstacle_list = [i for j, i in enumerate(self.cur_obstacle_list) if j not in remove_set] 

        # Refill obstacle list if possible
        for i in range(len(self.cur_obstacle_list), self.obstacle_list_size):
            index_list.append(self.add_obstacle())

        return index_list
        

    def add_obstacle(self):
        '''
        Add box to obstacle tracking list
        '''
        index = self.min()
        if index >= 0:
            self.cur_obstacle_list.append(copy.deepcopy(self.boxes[index]))
        return index

    def min(self):
        '''
        Finds the object with the closest distance to the middle which label is not already in the obstacle list
        '''

        min_distance = 10000000
        min_index = -1
        for i in range(len(self.boxes)):
            box = self.boxes[i]
            if (box.dist < min_distance) and (box not in self.cur_obstacle_list):
                min_distance = box.dist
                min_index = i
        
        return min_index


    def get_size_ratio_list(self):
        '''
        Returns a list of size ratios of the objects in obstacle tracking list
        '''

        ratio_list = []

        for box in self.cur_obstacle_list:
            
            ratio = box.size / box.old_size if box.old_size > 0 else 0
            ratio_list.append(ratio)
            box.old_size = box.size

        return ratio_list


    def reset(self):
        '''
        Clear obstacle tracking list
        '''
        self.cur_obstacle_list = []


    def clear(self):
        '''
        Empty detected boxes list
        '''
        self.boxes = []


class Box_Properties:
    '''
    Holds property value for a detected object
    '''

    def __init__(self, coords, size, middle, dist, label, score):
        self.coordinates = coords
        self.middle = middle
        self.size = size
        self.dist = dist
        self.label = label
        self.score = score
        self.old_size = 0


    def __eq__(self, other):
        '''
        Equal if the label is the same and middle coordinate within radius of 30 pixels
        '''
        return self.label == other.label and distance(self.middle, other.middle) < 30


    def __str__(self):
        return str(self.label) + ": " + str(self.score) + "%, " + str(self.coordinates)