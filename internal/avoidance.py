import logging
import copy
import time
from . import waypoint as wp
from . import position

log = logging.getLogger(__name__)


OBSTACLE_DISTANCE = 3.0
STOP_ON_DETECT = False

class Avoidance:
    """
    Class for avoiding obstacles
    Uses the image zones calculated by the detector to send the correct
    movement commands to the drone
    """

    def __init__(self, drone, logfile=None, log_level = logging.DEBUG, enabled=False):
        self.enabled = enabled
        self.drone = drone
        self.debug = False
        self.xmargin = None
        self.ymargin = None
        self.avoid_amount = 0.20
        self.avoidance_speed = 0.5
        self.logfile = logfile
        log.setLevel(log_level)


    def calculate_direction(self, zones):
        """
        Compares image zones to set horizontal/vertical movement values
        """
        
        if zones.l <= 0 and zones.r <= 0: # No free-zones right and left
            move_hor = 0
        elif zones.l == zones.r: # Left and right zones are equal
            move_hor = 1 # Move left for now --- TODO change this depending on waypoint coordinates
        else:
            if zones.l > zones.r: # Left zone bigger than right zone
                move_hor = 1
            else: # Right zone bigger than Left zone
                move_hor = 2

        if zones.u <= 0 and zones.d <= 0: # No free-zones up and down
            move_ver = 0
        elif zones.u == zones.d: # Up and down zones are equal
            move_ver = 1 # Move up for now --- TODO change this depending on waypoint coordinates
        else:
            if zones.u > zones.d: # Up zone bigger than down
                move_ver = 1
            else: # Down zone bigger than up
                move_ver = 2

        return move_hor, move_ver


    def calculate_control(self, move_hor, move_ver):
        """
        Calculates the waypoint for the avoidance controller based on move_hor and move_ver values
        """

        throttle = 0
        roll = 0

        if self.drone is not None:
            pos = copy.deepcopy(self.drone.pos)
        else:
            pos = position.Position(0,0,0)

        if self.drone is not None and move_ver == 0 and move_hor == 0:
            self.drone.wp_controller.disable()
            self.drone.halt()
            return

        command = "Going: "
        
        if move_ver == 1:
            pos.y += self.avoid_amount
            throttle = self.avoidance_speed
            command += "Up"
        elif move_ver == 2:
            pos.y -= self.avoid_amount
            throttle = -self.avoidance_speed
            command += "Down"
        command += " | "
        if move_hor == 1:
            pos.z -= self.avoid_amount
            roll = -self.avoidance_speed
            command += "Left"
        elif move_hor == 2:
            pos.z += self.avoid_amount
            roll = self.avoidance_speed
            command += "Right"

        log.debug(command)
        
        if self.drone is not None:
            self.drone.wp_controller.disable()
            log.info("Cur drone pos: " + str(self.drone.pos))
            self.drone.wp_avoid.set_waypoint(throttle, roll)


    def avoid(self, stop, zones, testmode=None):
        """
        Function called by the detector to avoid an obstacle
        """

        if not self.enabled:
            return

        self.enabled = False

        if testmode == 'ratio':
            self.drone.halt()
            with open(self.logfile, "a") as f:
                f.write(str(OBSTACLE_DISTANCE - self.drone.pos.x) + "\n")
            self.drone.return_home()
        else:

            if not stop and not STOP_ON_DETECT: # Obstacle can still be avoided
                assert zones is not None
               
                log.info("Avoid obstacle")

                # Move in the horizontal/vertical direction
                # 0 = no movement, 1 = move left/up, 2 = move right/down
                move_hor, move_ver = self.calculate_direction(zones)

                self.calculate_control(move_hor, move_ver)

            elif self.drone is not None:
                log.info("Halt drone")
                self.drone.halt() 
                self.drone.return_home()


    def enable(self):
        self.enabled = True


    def set_debug_mode(self, to):
        self.debug = to