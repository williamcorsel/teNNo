import cv2

class Hud:
    """
    Class for writing various hud elements
    """


    def write_text(self, frame, x, y, text, size=1.0, color=(255,0,0)):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, size, color)

    def write_warning(self, frame, x, y):
        self.write_text(frame, x, y, "!", 4.0, (0,0,255))
        