import numpy as np
import cv2 as cv
import argparse

cap = cv.VideoCapture("input.mp4")
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.05,
                       minDistance = 5,
                       blockSize = 2 ,
                       useHarrisDetector = True)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (35, 35),
                  maxLevel = 3,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0,255,(100,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc('M','P','4','V'), 10, (frame_width,frame_height))
noFrame = 0
while(1):
    noFrame+=1
    if(noFrame == 50):
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        noFrame = 0
    ret,frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv.add(frame,mask)
    cv.imshow('frame',img)
    out.write(img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
cap.release()
out.release()
