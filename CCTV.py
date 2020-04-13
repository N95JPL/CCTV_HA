from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import json
from datetime import datetime
from datetime import timedelta
import imutils
import cv2
import time
import os
from kcw.keyclipwriter import KeyClipWriter
import sys

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# for i in range(0, 100):
#         time.sleep(0.1)
#         sys.stdout.write(u"\u001b[1000D" + str(i + 1) + "%")
#         sys.stdout.flush()
print("[START] JPLogics - Home Protection")
print("[INFO] Kicking off script - " +
        datetime.now().strftime("%H:%M:%S=%d-%m-%Y"))
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
args = vars(ap.parse_args())
conf = json.load(open(args["conf"]))

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
#fps = FPS().start()
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    "prototxt/MobileNetSSD_deploy.prototxt.txt", "caffemodel/MobileNetSSD_deploy.caffemodel")
camera = cv2.VideoCapture(conf["ip_cam_addr"],cv2.CAP_FFMPEG)
#fps = camera.get(cv2.CAP_PROP_FPS)
kcw = KeyClipWriter(bufSize=conf["bufferSize"])
ret, frame = camera.read()
mask = np.zeros(frame.shape,dtype=np.uint8)
#roi_corners = np.array([[(0,0),(650,0), (1420,480),(1800,1080), (0,1080)]], dtype=np.int32)
roi_corners = np.array([[(237,0),(619,3),(745,114),(862,86),(1016,178),(1111,236),(1114,122),(1323,229),(1289,373),(1393,454),(1396,471),(1379,480),(1348,646),(1779,1078),(531,1075),(490,927),(399,721),(288,315)]], dtype=np.int32)
#(0,0),(611,0),(779,149),(897,125),(1016,175),(1111,236),(1118,124),(1301,222),(1287,371),(1393,454),(1348,644),(1781,1078),(0,1078),(0,276)
channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your image
ignore_mask_color = (255,)*channel_count
consecFrames = 0
systemActive = 0
print("[INFO] warming up...")
avg = None
lastUploaded = datetime.now()
(h, w) = (None, None)
person_number = 0
camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)
detectframe = datetime.now()
person_detected_image = False
person_recording = 0
nonMotionFrames = 0
# start = time.time()
# # for i in range(0, 60) :
# #     ret, frame = camera.read()
# #     kcw.update(frame)
# end = time.time()
# seconds = end - start
# fps = 60 / seconds
# print("[TEST] FPS Time: " + str(seconds))
# print("[TEST] FPS: " + str(fps))
while(1):
    conf = json.load(open(args["conf"]))
    ret, frame = camera.read()
    timestamp = datetime.now()
    if ret == False:
        print("[ERROR] Frame is empty. Restarting at: " +
              datetime.now().strftime("%H:%M:%S=%d-%m-%Y"))
        camera.release()
        cv2.destroyAllWindows()
        if person_recording > 0:
            kcw.finish()
        time.sleep(1)
        camera = cv2.VideoCapture(conf["ip_cam_addr"], cv2.CAP_FFMPEG)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 20)
        rebootTrigger = datetime.now() + timedelta(minutes=10)
    else:
        (h, w) = frame.shape[:2]
        if detectframe <= timestamp:
            #fps = camera.get(cv2.CAP_PROP_FPS)
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            masked_frame = cv2.bitwise_and(frame, mask)
            detectframe = datetime.now() + timedelta(seconds=conf["detectTime"])
            person_number = 0
            blob = cv2.dnn.blobFromImage(cv2.resize(masked_frame, (300, 300)),
                                         conf["scale"], (300, 300), 62)
            #0.007843
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = "{}: {:.2f}%".format(CLASSES[idx],
                                                 confidence * 100)
                    if "person" in label:
                        consecFrames = 0
                        person_number = person_number+1
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)
        # COLORS[idx],
                        #y = startY - 15 if startY - 15 > 15 else startY + 15
                        #cv2.putText(frame, label, (startX, y),
                                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if person_number == 0:
                nonMotionFrames += 1
            else:
                nonMotionFrames = 0
        cv2.putText(frame, conf["camera_name"], (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        if person_number > 0:
            cv2.putText(frame, 'Recording...', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not person_detected_image:
                if conf["dayFolder"]:
                    ts = timestamp.strftime("%H:%M:%S")
                    filename = (conf["person_image_file"] + "-" + ts)
                    if not os.path.exists(conf["userImageDir"] +
                                timestamp.strftime("%d-%m-%Y/")):
                        os.makedirs(conf["userImageDir"] +
                                timestamp.strftime("%d-%m-%Y/"))
                    image_path = (conf["userImageDir"] +
                                timestamp.strftime("%d-%m-%Y/") +
                                "/{filename}.jpg").format(filename=filename)
                else:
                    ts = timestamp.strftime("%H:%M:%S=%d-%m-%Y")
                    filename = (conf["person_image_file"] + "-" + ts)
                    image_path = (conf["userImageDir"] +
                                "/{filename}.jpg").format(filename=filename)
                cv2.imwrite(image_path, frame)
                person_detected_image = True
            if not person_recording:
                person_recording = 1
                if conf["dayFolder"]:
                    ts = timestamp.strftime("%H:%M:%S")
                    filename = (conf["person_image_file"] + "-" + ts)
                    if not os.path.exists(conf["userVideoDir"] +
                                timestamp.strftime("%d-%m-%Y/")):
                        os.makedirs(conf["userVideoDir"] +
                                timestamp.strftime("%d-%m-%Y/"))
                    video_path = (conf["userVideoDir"] +
                                timestamp.strftime("%d-%m-%Y/") +
                                "/{filename}.avi").format(filename=filename)
                else:
                    ts = timestamp.strftime("%H:%M:%S=%d-%m-%Y")
                    filename = (conf["person_image_file"] + "-" + ts)
                    video_path = (conf["userVideoDir"] +
                                "/{filename}.avi").format(filename=filename)
                kcw.start(video_path, cv2.VideoWriter_fourcc(
                    'M', 'J', 'P', 'G'), 16)
                ts = timestamp.strftime("%H:%M:%S=%d-%m-%Y")
                print("[DETECT] " + label + " detected @ " + ts)

        #cv2.putText(frame, ("FPS:"+str(fps)), (20, 80),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        frameResize = imutils.resize(frame, width=conf["resizeWidth"])
        #cv2.namedWindow("Camera", cv2.CV_GUI_EXPANDED)
        cv2.imshow("Camera", frameResize)
        #masked_frame1 = imutils.resize(masked_frame, width=conf["resizeWidth"])
        #(h, w) = frame.shape[:2]
        #cv2.imshow("Camera-MBeta", masked_frame1)
        
        if person_number == 0 and person_recording > 0:
            if nonMotionFrames > 2:
                if person_recording == conf["bufferSize"]:
                    cv2.putText(frame, 'Recording...', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    kcw.update(frame)
                    kcw.finish()
                    person_recording = False
                    person_detected_image = False
                    ts = timestamp.strftime("%H:%M:%S=%d-%m-%Y")
                    print("[DETECT] Ended @ " + ts)
                else:
                    cv2.putText(frame, 'Recording...', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    person_recording += 1
            else:
                cv2.putText(frame, 'Recording...', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        kcw.update(frame)
        #print("")
        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):
            person_number = 1
            label = "test"
        if key == ord('s'):
            person_number = 0
        if key == ord('q'):
            kcw.finish()
            break

camera.release()
cv2.destroyAllWindows()

#(0,0),(611,0),(779,149),(897,125),(1016,175),(1111,236),(1118,124),(1301,222),(1287,371),(1393,454),(1348,644),(1781,1078),(0,1078),(0,276)
#(237,0),(619,3),(745,114),(862,86),(1016,178),(1111,236),(1114,122),(1323,229),(1289,373),(1393,454),(1396,471),(1379,480),(1348,646),(1779,1078),(531,1075),(490,927),(399,721),(288,315)