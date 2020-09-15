import time

class Fps:
    '''
    Class for tracking the average FPS of the player
    '''

    def __init__(self, average_of=10):
        self.number_frames = 0
        self.average_of = average_of
        self.fps = 0
        self.start = 0


    def update(self):
        '''
        Update FPS value. calculate average every self.average_of frames and count frame
        '''
        if self.number_frames % self.average_of == 0:
            if self.start != 0:
                self.fps = self.average_of / (time.time() - self.start)
                
            self.start = time.time()
        self.number_frames += 1


    def get(self):
        return self.fps
