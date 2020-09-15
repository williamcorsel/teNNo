import time
import copy
import logging
import datetime

import tellopy
import av
import cv2
from simple_pid import PID

import internal.position as pos
import internal.waypoint as wp
import internal.waypoint_controller as wpc


log = logging.getLogger(__name__)


class Drone:
    """
    Class used to control various drone functions such as
    - Initial setup
    - Video streaming
    - Logging
    - Controls
    """

    def __init__(self):
        self.tello = tellopy.Tello()
        self.container = None # Video frames
        self.ref_pos = None # Reference position to account for "random" MVO values on take off
        self.ab_pos = pos.Position(0,0,0) # Absolute position output by drone
        self.pos = pos.Position(0,0,0) # Position relative to ref_pos
        self.waypoint = wp.Waypoint(pos.Position(2.8, 0, 0)) # Waypoint to the drone to fly to relative to (0,0,0)
        self.wp_controller = wpc.Waypoint_Controller(self, self.waypoint, False) # Controls based on distance to waypoint
        self.wp_avoid = wpc.Avoidance_Controller(self, None, False, continue_after=True) # Controls based on distance to waypoint set by obstacle avoider
        self.battery = -1 # battery level
        self.record = False
        self.running = True

        self.logging = False
        self.log_file = None # log file object
        self.log_file_path = None # log file path
        self.start_time = None # For measuring time since start experiment run
            

    def connect(self):
        """
        Connects the drone and blocks until a connection is established
        """

        self.tello.connect()
        self.tello.wait_for_connection(60.0)

    def log_data(self):
        """
        Enables the logging of data
        """

        assert self.tello is not None
        self.tello.subscribe(self.tello.EVENT_LOG_DATA, self.log_handler)
        self.tello.subscribe(self.tello.EVENT_FLIGHT_DATA, self.log_handler)
        self.tello.subscribe(self.tello.EVENT_FILE_RECEIVED, self.log_handler)


    def log_handler(self, event, sender, data):
        """
        Handler for data logging. Updates position of the drone
        """
    
        if event is sender.EVENT_LOG_DATA:
            self.ab_pos.x = x = data.mvo.pos_x
            self.ab_pos.y = y = data.mvo.pos_y
            self.ab_pos.z = z = -data.mvo.pos_z

            # Update coordinates
            if abs(x) + abs(y) + abs(z) > 0.07:
                if self.ref_pos is None:
                    self.ref_pos = pos.Position(x, y, z) # Set initial reference position
                else:
                    self.pos.x = x - self.ref_pos.x
                    self.pos.y = y - self.ref_pos.y
                    self.pos.z = z - self.ref_pos.z

            # Write to log file if enabled
            if self.logging:
                elapsed_time = time.time() - self.start_time  
                self.log_file.write(str(elapsed_time) + "," + str(self.pos) + ",%s\n" % data.format_cvs())
                

        if event is sender.EVENT_FLIGHT_DATA:
            self.battery = data.battery_percentage



    def init_video(self):
        """
        Enables video streaming. Results will be stored in self.container
        """

        assert self.container is None

        retry = 3
        while self.container is None and 0 < retry:
            retry -= 1
            try:
                self.container = av.open(self.tello.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')


        assert self.container is not None

        
    def set_position(self, pos):
        """
        Sets the reference position of the drone
        """
        self.ref_pos = pos

    def reset_position(self):
        """
        Resets the reference position to the drone's position -> relative position will be (0,0,0)
        """
        self.set_position(copy.deepcopy(self.ab_pos))    

    def reset_waypoint(self):
        '''
        Reset waypoint to value stored in self.waypoint
        '''
        self.wp_controller.set_waypoint(self.waypoint)

    def control(self):
        """
        Chooses to let the avoider or waypoint controller send commands to the drone
        """
        if self.wp_avoid.enabled:
            self.wp_avoid.pid_control()
        elif self.wp_controller.enabled:
            self.wp_controller.pid_control()


    def toggle_recording(self, to):
        self.record = to


    def set_logging(self, to_on, path=None):
        if to_on:
            # Create new log file
            self.log_file_path = path
            self.log_file = open(path, 'w+')
            self.start_time = time.time()
            self.logging = True
        else:
            self.logging = False
            self.log_file = None
    

    def toggle_logging(self):
        if self.logging:
            self.set_logging(False)
        else:
            path = './flight_logs/tello-%s.csv' % (datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            self.set_logging(True, path)
           

    def return_home(self):
        '''
        Stop run and return to 0 coordinate
        '''
        self.wp_avoid.disable()
        self.wp_controller.disable()
        self.halt()
        self.wp_controller.set_waypoint(wp.Waypoint(pos.Position(0, 0, 0)))
        self.wp_controller.enable()


    def quit(self):
        self.running = False

    """
    Movement commands
    """

    def move(self, speed, direction):
        '''
        Move function to interrupt any autonomous operation
        '''
        self.wp_controller.disable()
        self.wp_avoid.disable()

        if direction == "forward":
            self.tello.forward(speed)
        elif direction == "backward":
            self.tello.backward(speed)
        elif direction == "left":
            self.tello.left(speed)
        elif direction == "right":
            self.tello.right(speed)
        elif direction == "up":
            self.tello.up(speed)
        elif direction == "down":
            self.tello.down(speed)
        elif direction == "clockwise":
            self.tello.clockwise(speed)
        elif direction == "counter_clockwise":
            self.tello.counter_clockwise(speed)
        else:
            log.error("Unknown direction option")


    def halt(self):
        """
        Stops all movement commands of the drone
        """
        log.info("Halting drone")
        self.tello.forward(0)
        self.tello.backward(0)
        self.tello.left(0)
        self.tello.right(0)
        self.tello.up(0)
        self.tello.down(0)

    # def left(self, distance):
    #     command = "left " + str(distance)
    #     log.info(command)
    #     self.tello.sock.sendto(command.encode("utf-8"), self.tello.tello_addr)

    # def right(self, distance):
    #     command = "right " + str(distance)
    #     log.info(command)
    #     self.tello.sock.sendto(command.encode("utf-8"), self.tello.tello_addr)

    # def up(self, distance):
    #     command = "up " + str(distance)
    #     log.info(command)
    #     self.tello.sock.sendto(command.encode("utf-8"), self.tello.tello_addr)

    # def down(self, distance):
    #     command = "down " + str(distance)
    #     log.info(command)
    #     self.tello.sock.sendto(command.encode("utf-8"), self.tello.tello_addr)

    # def go(self, x, y, z, speed):
    #     """
    #     TODO: remove
    #     """
    #     command = "go " + str(x) + " " + str(y) + " " + str(z) + " " + str(speed)
    #     log.info(command)
    #     self.tello.sock.sendto(command.encode("utf-8"), self.tello.tello_addr)

    def takeoff(self):
        self.tello.takeoff()
        #self.wp_controller.enabled = True


    def land(self):
        self.wp_controller.enabled = False
        self.tello.land()


    def set_throttle(self, value):
        self.tello.set_throttle(value)


    def set_roll(self, value):
        self.tello.set_roll(value)


    def set_pitch(self, value):
        self.tello.set_pitch(value)


if __name__ == '__main__':
    drone = Drone()
    drone.log_data()

    
