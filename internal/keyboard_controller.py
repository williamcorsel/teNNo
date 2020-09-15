"""
Controls:
see README
"""

from pynput import keyboard
import datetime

class Keyboard_Controller:
    """
    Class for enabling keyboard input for the drone

    Attributes
    ----------
    drone : drone.Drone
        The drone to control
    keydown : bool
        True if a key is pressed down, False if no keys are pressed
    speed : float
        Speed of the drone operations
    """

    def __init__(self, drone, player, detector=None, testmode=None, recordmode=False):
        """
        Initialization function

        Parameters
        ----------
        drone : drone.Drone
            Drone object which should already have been initialized
        """

        assert drone is not None
        
        self.drone = drone
        self.player = player
        self.detector = detector
        self.keydown = False
        self.speed = 50.0
        self.testmode = testmode
        self.recordmode = recordmode
        

    def on_press(self, keyname):
        """
        Handler for keyboard listener. Called when a button is pressed down
        
        Parameters
        ----------
        keyname : pynput.keyboard.Key
            name of the key pressed returned by the Listener
        """

        if self.keydown: # If a key is already pressed down, return
            return
        try:
            self.keydown = True
            keyname = str(keyname).strip("u'") # strip these chars to isolate the button pressed
            print('+' + keyname)
            if keyname == 'Key.esc':
                self.drone.quit()
                return
            elif keyname == 'o':
                if self.detector != None:
                    self.detector.toggle()
                return
            elif keyname == 'l':
                self.drone.toggle_logging()
                return
            elif keyname == 'p' or keyname == '1':
                self.drone.reset_position()
                return
            elif keyname == 'h':
                self.drone.return_home()
                self.drone.set_logging(False)
                return
            elif keyname == '-':
                self.player.toggle_hud()
            elif keyname == 'Key.home' and self.testmode == 'avoid':
                self.detector.write_to_log("OK\n")
                self.drone.set_logging(False)
                if self.recordmode:
                    self.drone.toggle_recording(False)
            elif keyname == 'Key.end' and self.testmode == 'avoid':
                self.detector.write_to_log("FAIL\n")
                self.drone.set_logging(False)
                if self.recordmode:
                    self.drone.toggle_recording(False)
            elif keyname == '2':
                self.drone.wp_avoid.disable()
                self.drone.wp_controller.toggle_enabled()
                return
            elif keyname == '3':
                if self.testmode == 'avoid':
                    path = './flight_logs/tello-%s.csv' % (datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
                    self.detector.write_to_log(path + ",")
                    self.drone.set_logging(True, path)
                    if self.recordmode:
                        self.drone.toggle_recording(True)

                if self.detector != None:
                    self.drone.reset_waypoint()
                    self.detector.enable()
                    self.detector.avoider.enable()
                    self.drone.wp_controller.enable()
                
                return 
            elif keyname == 'm':
                if self.detector is not None:
                    self.detector.add_ratio(0.025)
            elif keyname == 'n':
                if self.detector is not None:
                    self.detector.add_ratio(-0.025)
            elif keyname == 'r':
                self.drone.toggle_recording(not self.drone.record)
                return
            elif keyname in self.controls:
                key_handler = self.controls[keyname]
                key_handler(self.speed)

        except AttributeError:
            print('special key {0} pressed'.format(keyname))


    def on_release(self, keyname):
        """
        Handler for keyboard listener. Called when a button is released.
        Stops the operation activated by the key by setting the speed to 0
        
        Parameters
        ----------
        keyname : pynput.keyboard.Key
            name of the key pressed returned by the Listener
        """

        self.keydown = False
        keyname = str(keyname).strip("u'") # strip these chars to isolate the button pressed
        print('-' + keyname)
        if keyname in self.controls:
            print("Found key in controls")
            key_handler = self.controls[keyname]
            key_handler(0)
    

    def init_controls(self):
        """
        Define keys and add listener
        """
        self.controls = {
            'w': lambda speed: self.drone.move(speed, "forward"),
            's': lambda speed: self.drone.move(speed, "backward"),
            'a': lambda speed: self.drone.move(speed, "left"),
            'd': lambda speed: self.drone.move(speed, "right"),
            'Key.up': lambda speed: self.drone.move(speed, "up"),
            'Key.down': lambda speed: self.drone.move(speed, "down"),
            'q': lambda speed: self.drone.move(speed, "counter_clockwise"),
            'e': lambda speed: self.drone.move(speed, "clockwise"),
            'Key.tab': lambda speed: self.drone.takeoff(),
            'Key.backspace': lambda speed: self.drone.land(),
        }

        self.key_listener = keyboard.Listener(on_press=self.on_press,
                                              on_release=self.on_release)
        self.key_listener.start()
        print("ENABLED KEYBOARD CONTROL")
        