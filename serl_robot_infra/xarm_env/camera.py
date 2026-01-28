import cv2
import time
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(3000):
    cam.grab()
    _, image = cam.retrieve()
    cv2.imshow('1', image)
    # cv2.imwrite("cam.png", image)    
    cv2.waitKey(1)
    time.sleep(0.01)
