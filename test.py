import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while 1:
    _, img = cap.read()
    cv2.imshow('w', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
