import cv2
import numpy as np

class BallTrackerRenderer:
    def __init__(self):
        pass

    def draw_rad(self, rad):
        # rad: (x, y), radius
        cv2.circle(self.frame, (int(rad[0][0]), int(rad[0][1])), int(rad[1]),
            (0, 255, 255), 2)
    
    def draw_center(self, center, pts):
        cv2.circle(self.frame, center, 5, (0, 0, 255), -1)
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(16 / float(i + 1)) * 2.5)
            cv2.line(self.frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    def draw(self, frame, center, pts, rad):
        self.frame = frame
        if rad[1] > 10:
            self.draw_rad(rad)
            self.draw_center(center, pts)

        return self.frame