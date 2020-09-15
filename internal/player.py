import cv2
import time
import internal.hud as hud
import internal.fps as fps
from datetime import datetime
import threading
import numpy

frame = None
run_recv_thread = True



class Player:
    """
    Video player class for showing drone video
    """

    def __init__(self, drone, detector, recordmode=False):
        self.drone = drone
        self.container = drone.container
        self.detector = detector
        self.height = 720
        self.width = 960
        self.fps_cap = 11

        self.recordmode = recordmode

        self.hud = hud.Hud()
        self.hud_enabled = True

        self.recorder = None
        self.fps = fps.Fps()


    def recv_thread(self, container):
        '''
        Separate thread to receive video frames and place them in the frame variable
        '''

        global frame
        global run_recv_thread

        try:
            while run_recv_thread:
                for f in container.decode(video=0):
                    frame = f
                time.sleep(0.01)
        except Exception as ex:
            print(ex)


    def show_video(self):
        """
        Shows video and handles control of the drone:
        - Display frames from frame variable set by recv_thread
        - Sends frames to obstacle detector if enabled
        - Calls drone control
        """

        global frame
        global run_recv_thread

        old = None
        period = 1.0 / self.fps_cap
       
        threading.Thread(target=self.recv_thread, args=(self.drone.container,)).start()

        start = time.time()
        while self.drone.running:
    
            if frame is None:
                time.sleep(0.01)
            else:
                if time.time() - start >= period:
                    old, output = self.detector.process_frame(frame, old)
                    output = cv2.resize(output, (self.width,self.height), cv2.INTER_AREA)
                    self.write_hud(output, self.fps.get())
                    
                    cv2.imshow("Video output", output)
                    cv2.waitKey(1)

                    self.fps.update()

                    # save frames if recording
                    if self.drone.record: 
                        if self.recorder is None: # first frame -> create new file
                            filename = "recordings/tello_recording-" + datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".avi"
                            self.recorder = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc('M','J','P','G'), self.fps.get(), (960,720))
                            if self.recordmode:
                                self.detector.write_to_log(filename + ',')
                        
                        self.recorder.write(output)
                    elif not self.drone.record and self.recorder is not None: # remove writer to old file
                        self.recorder = None

                frame = None # clear frame and wait for the next
            
            # Movement commands to drone
            self.drone.control()
            
        # Shut off recv_thread    
        run_recv_thread = False


    def toggle_hud(self):
        self.hud_enabled = not self.hud_enabled


    def write_hud(self, frame, fps):
        """
        Writes hud to frame
        """
        if not self.hud_enabled:
            return

        if self.detector.obstacle_detected:
            self.hud.write_warning(frame, self.width - 30, 60)
            self.detector.obstacle_detected = False

        self.hud.write_text(frame, 10, self.height - 10, str(self.drone.pos))
        self.hud.write_text(frame, 10, self.height - 25, str(self.drone.battery))
        self.hud.write_text(frame, 10, 15, str(round(fps)))

        if self.detector.testmode == 'ratio_size':
            self.hud.write_text(frame, self.width - 50, 20, str(self.detector.hull_threshold))
        elif self.detector.testmode == 'ratio_kp':
            self.hud.write_text(frame, self.width - 50, 20, str(self.detector.kp_threshold))


    def show_possible_obstacle(self):
        """
        DEBUG: show frames with highest hull/kp ratio
        """
        print("Max hull ratio: " + str(self.detector.max_hull_ratio))
        print("Max kp ratio: " + str(self.detector.max_kp_ratio))
        cv2.imshow("Max prev", self.detector.max_previous)
        cv2.imshow("Max cur", self.detector.max_current)
        cv2.waitKey(0)
