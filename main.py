import cv2
import numpy as np
import time
# import copy
import math
import os

### TODO: implement clustering and calculating inangle as angle inside a finger


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (40, 40)
fontScale              = 0.6
fontColor              = (0, 0, 0)
lineType               = 2


kernel_elliptic = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9))


cap = cv2.VideoCapture(0)


width = int(cap.get(3))  # float
height = int(cap.get(4)) # float

x0 = int(width / 2)
y0 = 0
x1 = width
y1 = int(0.75 * height)


cap_brightness = 0.5
while 1:
    cap.set(10, cap_brightness)
    ave_v = np.average(np.hstack(cv2.flip(cap.read()[1], +1)[y0:y1,x0:x1][:,:,2]))
    print(ave_v)
    os.system("echo $(v4l2-ctl --get-ctrl=brightness)")
    print('cap_brightness:', cap_brightness)

    if ave_v < 130:
        cap_brightness += 0.05
        continue

    if ave_v > 140:
        cap_brightness -= 0.05
        continue

    break

def findfingers(target_frame, mask):
    # target_frame = frame.copy()

    blurred = cv2.medianBlur(mask, 5)
    # _, blurred = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)


    eroded = cv2.erode(blurred, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9, 9)), 2)
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 3)

    cv2.imshow('b', dilated)

    _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    try:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        # cv2.drawContours(target_frame, [biggest_contour], -1,(30,30,255),3)

        hull = cv2.convexHull(biggest_contour, returnPoints = False)
        # cv2.drawContours(target_frame, [hull], -1,(30,30,255),3)

        bx, by, bw, bh = cv2.boundingRect(biggest_contour)
        # cv2.rectangle(target_frame, (bx,by), (bx+bw, by+bh) , [255, 0, 0], 3)
        cx = bx + int(bw / 2)
        cy = by + int(bh / 2)

        count = 0

        if len(hull) > 3:

            defects = cv2.convexityDefects(biggest_contour, hull)

            if type(defects) != type(None):  # avoid crashing.   (BUG not found)

                for i in range(defects.shape[0]):
                    s, e, f, _ = defects[i,0] # (s, e, f, d)
                    start = tuple(biggest_contour[s][0])
                    end = tuple(biggest_contour[e][0])
                    far = tuple(biggest_contour[f][0])

                    cv2.line(target_frame,start,far,[0,255,0],2)
                    cv2.line(target_frame,far,end,[0,255,0],2)
                    cv2.circle(target_frame,far,5,[0,0,255],-1)

                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    inangle = abs(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))  # cosine theorem

                    c_angle = cv2.fastAtan2(cy - start[1], cx - start[0]) #* 180 / math.pi
                    # fastAtan2 returns degrees
                    dlength = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)

                    if inangle <= (math.pi * 2 / 3) and inangle >= (20 / 180 * math.pi) and c_angle > -30 and c_angle < 160 and dlength > 0.1 * bh:  # inangle less than 90 degree, treat as fingers
                        count += 1
                        cv2.circle(target_frame, far, 8, [211, 84, 0], -1)

                    # if count == 0:


        cv2.putText(target_frame, str(count + 1), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

    except ValueError:
        pass

    cv2.rectangle(target_frame, (0,0), (x0, y1), (0, 255, 0), 3)

    # return target_frame

cap.set(cv2.CAP_PROP_CONTRAST, 0.5)

while cap.isOpened():

    _, frame = cap.read()
    frame = cv2.flip(frame, +1)
    target_frame = frame[y0:y1,x0:x1]
    cv2.rectangle(target_frame, (0,0), (x0, y1), (0, 255, 0), 3)
    cv2.imshow('frame', frame)

    hsv = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,25,20]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165,15,20]), np.array([180, 255, 255]))
    hsv_mask = cv2.bitwise_or(mask1, mask2)

    findfingers(target_frame=target_frame, mask=hsv_mask)
    cv2.imshow('frame',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break



cap.release()
cv2.destroyAllWindows()
