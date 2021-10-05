import cv2
import imutils
import depthai as dai
from collections import deque

class BallTracker:
    def __init__(self):
        self.redLower = (114, 72, 0)
        self.redUpper = (176, 208, 218)
        self.pts = deque(maxlen=16)


    def create_pipeline(self):
        pass

    def recognize_ball(self, frame):
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "red", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.redLower, self.redUpper)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        self.center = None
        self.rad = ((0, 0), 0)

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                self.center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # update the points queue
                    self.rad = ((x, y), radius)
                    self.pts.appendleft(self.center)


    def next_frame(self, frame):
        self.recognize_ball(frame)
        return self.center, self.pts, self.rad