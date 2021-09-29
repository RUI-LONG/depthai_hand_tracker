# USAGE
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import depthai as dai


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "red"
# ball in the HSV color space, then initialize the
# list of tracked points
# redLower = (112, 6, 22)
# redUpper = (179, 255, 38)

# blueLower = (110, 80, 0)
# blueUpper = (160, 186, 200)
def nothing(x):
    pass
cv2.namedWindow('Frame')

cv2.createTrackbar('HMin','Frame',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','Frame',0,255,nothing)
cv2.createTrackbar('VMin','Frame',0,255,nothing)
cv2.createTrackbar('HMax','Frame',0,179,nothing)
cv2.createTrackbar('SMax','Frame',0,255,nothing)
cv2.createTrackbar('VMax','Frame',0,255,nothing)

blueLower = (80, 80, 0)
blueUpper = (160, 186, 200)

cv2.setTrackbarPos('HMin', 'Frame', 80)
cv2.setTrackbarPos('SMin', 'Frame', 80)
cv2.setTrackbarPos('VMin', 'Frame', 0)

cv2.setTrackbarPos('HMax', 'Frame', 160)
cv2.setTrackbarPos('SMax', 'Frame', 186)
cv2.setTrackbarPos('VMax', 'Frame', 200)


pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(600, 600)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# keep looping
with dai.Device(pipeline) as device:
    # grab the current frame
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        frame = inRgb.getCvFrame()
        # color space
        # frame = imutils.resize(frame, width=600)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "red", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        hMin = cv2.getTrackbarPos('HMin','Frame')
        sMin = cv2.getTrackbarPos('SMin','Frame')
        vMin = cv2.getTrackbarPos('VMin','Frame')

        hMax = cv2.getTrackbarPos('HMax','Frame')
        sMax = cv2.getTrackbarPos('SMax','Frame')
        vMax = cv2.getTrackbarPos('VMax','Frame')
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])
        mask = cv2.inRange(hsv, lower, upper)
        # mask = cv2.erode(mask, None, iterations=2)
        # mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                # only proceed if the radius meets a minimum size
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    cv2.circle(frame, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # update the points queue
        pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()