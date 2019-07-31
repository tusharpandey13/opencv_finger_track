import cv2
import numpy as np
import time
# import copy
import math
import os


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (40, 40)
fontScale              = 0.6
fontColor              = (0, 0, 0)
lineType               = 2


# open_palm_cascade = cv2.CascadeClassifier('open_palm.xml')


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
    # print(ave_v)
    # os.system("echo $(v4l2-ctl --get-ctrl=brightness)")
    # print('cap_brightness:', cap_brightness)

    if ave_v < 130:
        cap_brightness += 0.05
        continue

    if ave_v > 150:
        cap_brightness -= 0.05
        continue

    break

# os.system("echo $(v4l2-ctl --get-ctrl=contrast)")
# cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
# os.system("echo $(v4l2-ctl --get-ctrl=contrast)")


def findfingers(target_frame, mask):
        # target_frame = frame.copy()

        blurred = cv2.medianBlur(mask, 5)
        # blurred = cv2.GaussianBlur(mask, (3, 3), 0)
        ## waste
        # _, blurred = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)


        tmp = 9
        # kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))
        kernel_elliptic = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(tmp, tmp))
        # kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

        iter = 5

        eroded = cv2.erode(blurred, kernel_elliptic, 2)
        # dilated = cv2.dilate(blurred, kernel_rect, iter)
        dilated = cv2.dilate(eroded, kernel_elliptic, iter)
        # dilated = cv2.dilate(blurred, kernel_cross, iter)
        # dilated = cv2.GaussianBlur(dilated, (3,3), 0)
        # dilated = cv2.medianBlur(dilated, 5)
        # cv2.imshow('dilated', dilated)


        #
        # edged = cv2.Canny(dilated, 30, 200)
        #
        _, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

        try:
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

            # mask = np.zeros(image.shape, np.uint8)
            cv2.drawContours(target_frame, [biggest_contour], -1,(30,30,255),3)

            hull = cv2.convexHull(biggest_contour, returnPoints = False)
            # cv2.drawContours(target_frame, [hull], -1,(30,30,255),3)

            bx, by, bw, bh = cv2.boundingRect(biggest_contour)
            # cv2.rectangle(target_frame, (bx,by), (bx+bw, by+bh) , [255, 0, 0], 3)
            cx = bx + int(bw / 2)
            cy = by + int(bh / 2)

            count = 0

            if len(hull) > 3:

                defects = cv2.convexityDefects(biggest_contour, hull)

    # cnt = biggest_contour : contour
    # hull = cv2.convexHull(cnt,returnPoints = False)
    # defects = cv2.convexityDefects(cnt,hull)
    #
    # convexityDefects returns an array where each row contains these values -
    # [ start point, end point, farthest point, approximate distance to farthest point ].
    # Remember first three values returned are indices of cnt.
    # So we have to bring those values from cnt.

                if type(defects) != type(None):  # avoid crashing.   (BUG not found)

                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i,0] # (s, e, f, d)
                        start = tuple(biggest_contour[s][0])
                        end = tuple(biggest_contour[e][0])
                        far = tuple(biggest_contour[f][0])

                        # cv2.line(target_frame,start,far,[0,255,0],2)
                        # cv2.line(target_frame,far,end,[0,255,0],2)

                        # cv2.circle(target_frame,far,5,[0,0,255],-1)

                        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                        inangle = abs(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))  # cosine theorem

                        c_angle = cv2.fastAtan2(cy - start[1], cx - start[0]) #* 180 / math.pi
                        # fastAtan2 returns degrees

                        dlength = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)

                        if inangle <= (math.pi * 2 / 3) and inangle >= (20 / 180 * math.pi) and c_angle > -30 and c_angle < 160 and dlength > 0.1 * bh:  # inangle less than 90 degree, treat as fingers
                            count += 1
                            cv2.circle(target_frame, start, 8, [211, 84, 0], -1)


            cv2.putText(target_frame, str(count), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        except ValueError:
            pass

        cv2.rectangle(target_frame, (0,0), (x0, y1), (0, 255, 0), 3)

        # return target_frame


_, tmpframe = cap.read()
tmpframe = cv2.flip(tmpframe, +1)[y0:y1,x0:x1]

while cap.isOpened():

    _, frame = cap.read()
    frame = cv2.flip(frame, +1)
    target_frame = frame[y0:y1,x0:x1]

    cv2.imshow('f', target_frame)
    cv2.imshow('1', tmpframe)
    # _, thresh = cv2.threshold(cv2.bitwise_xor(tmpframe, target_frame), 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow('xor', thresh)
    cv2.imshow('-', tmpframe - target_frame)

    # target_frame.shape

    # palms = open_palm_cascade.detectMultiScale(frame, 1.3, 4)
    # # for (x, y, w, h) in palms:

    # for (x, y, w, h) in palms:
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # if len(palms) == 1:
    #     x1 = [int(width / 2), width][int(palms[0][0] / int(width / 2))]
    #     x0 = [0, int(width / 2)][int(palms[0][0] / int(width / 2))]

    # hsv1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(target_frame, cv2.COLOR_BGR2HSV)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # np.array([])
    # [0, 70, 40], [10, 170, 255]
    # [165, 70, 40], [180, 170, 255]
    #
    mask1 = cv2.inRange(hsv, np.array([0,25,20]), np.array([20, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165,15,20]), np.array([180, 255, 255]))
    hsv_mask = cv2.bitwise_or(mask1, mask2)
    # cv2.imshow('hsvmask', hsv_mask)

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, gray_mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # gray_mask = gray_mask[y0:y1,x0:x1]
    # cv2.imshow('graymask', gray_mask)

    # mask = cv2.bitwise_or(hsv_mask, gray_mask)
    # tf_at = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)

    # res = cv2.bitwise_and(frame,frame, mask= mask)

    # hsv_channels = cv2.split(res)





    # cv2.imshow('frame',findfingers(frame=target_frame, mask=mask))
    # cv2.imshow('hsvframe',findfingers(frame=target_frame, mask=hsv_mask))
    # cv2.imshow('grayframe',findfingers(frame=target_frame, mask=gray_mask))

    # cv2.imshow('gray', thresh)
    # cv2.imshow('mask1', mask1)
    # cv2.imshow('mask2', mask2)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', tf_at)
    # cv2.imshow('dilated', dilated)
    # cv2.imshow('blurred', blurred)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('b'):  # press 'b' to capture the background
        isBgCaptured = 1
        _, tmpframe = cap.read()
        tmpframe = cv2.cvtColor(cv2.flip(tmpframe, +1)[y0:y1,x0:x1], cv2.COLOR_BGR2GRAY)
        print( '!!!Background Captured!!!')
    elif k == ord('r'):  # press 'r' to reset the background
        isBgCaptured = 0
        tmpframe = None
        print ('!!!Reset BackGround!!!')


cap.release()
cv2.destroyAllWindows()
# print(tmpl1)
# np.average(tmpl1)
