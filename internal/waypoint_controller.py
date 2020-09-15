from simple_pid import PID
import logging
import time

log = logging.getLogger("WC")

forward_speed = 0.5

class Waypoint_Controller(object):
    '''
    Waypoint controller uses PID control to steer drone towards the supplied coordinates
    '''
    def __init__(self, drone, waypoint=None, enabled=False, ffs=False):
        self.drone = drone
        self.pid_roll = PID(0.5,0.00001,0.01,setpoint=0,output_limits=(-0.5,0.5))
        self.pid_throttle = PID(0.5,0.00001,0.01,setpoint=0,output_limits=(-0.5, 0.5))
        self.pid_pitch = PID(0.5,0.00001,0.01,setpoint=0,output_limits=(-0.5, 0.5))
        self.waypoint = waypoint
        self.enabled = enabled
        self.fixed_forward_speed = ffs

    def pid_control(self):
        if not self.enabled:
            return

        # If waypoint reached halt the drone
        if self.waypoint.reached(self.drone.pos): 
            self.disable() 
            self.drone.halt()    
            return

        zoff = self.drone.pos.z - self.waypoint.pos.z
        yoff = self.drone.pos.y - self.waypoint.pos.y
        xoff = self.drone.pos.x - self.waypoint.pos.x

        zerr = self.pid_throttle(zoff)
        yerr = self.pid_roll(yoff)

        if self.fixed_forward_speed:
            if self.drone.pos.x < self.waypoint.pos.x - 0.1:
                xerr = forward_speed
            else:
                xerr = 0
        else:
            xerr = self.pid_pitch(xoff)

        # Set drone movement values
        self.drone.set_throttle(zerr)
        self.drone.set_roll(yerr)
        self.drone.set_pitch(xerr)


    def toggle_enabled(self):
        if self.enabled:
            self.disable()
            self.drone.halt()
        else:
            self.enable()


    def enable(self):
        self.enabled = True


    def disable(self):
        if self.enabled:
            self.drone.halt()
        self.enabled = False
        

    def set_waypoint(self, waypoint):
        log.info("Setting waypoint to " + str(waypoint))
        self.waypoint = waypoint
        

class Avoidance_Controller(Waypoint_Controller):
    '''
    Controller for avoidance manoeuvres derived from Waypoint_Controller
    '''

    def __init__(self, drone, waypoint=None, enabled=False, ffs=True, continue_after=False):
        self.drone = drone
        self.waypoint = waypoint
        self.enabled = enabled
        self.fixed_forward_speed = ffs
        self.set_time = None
        self.throttle = None
        self.roll = None
        self.duration = 0.8
        self.continue_after = continue_after


    def pid_control(self):
        if not self.enabled:
            return

        if time.time() - self.set_time > self.duration:
            self.disable()

            # If continue to waypoint is enabled
            if self.continue_after:
                self.drone.wp_controller.enable()
            return


    def set_waypoint(self, throttle, roll):
        #log.info("Setting waypoint to " + str(waypoint))
        self.set_time = time.time()
        self.drone.tello.set_pitch(0.05)
        self.drone.tello.set_throttle(throttle)
        self.drone.tello.set_roll(roll)
        self.enable()