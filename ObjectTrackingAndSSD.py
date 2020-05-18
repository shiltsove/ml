import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
     "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
MIN_MATCH_COUNT = 4
option = 3 #SSD detected region
print("[INFO] loading model…")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
#tracker initiation
tracker = cv2.TrackerKCF_create()
initBB = None

capture = cv2.VideoCapture("InputVideos/car_video.avi")#Short video for testing


coordinates = []
# Radius of circle 
radius = 17  
# Blue color in BGR 
color = (255, 0, 0)  
# Line thickness of 2 px 
thickness = -1

img_count = -1
img_type = ".png"
conf_threshold = 0.6
padding = 10
car_outputs = "Car"
noCar_outputs = "DontCare"

cars = 0
m1 = 0
mask1 = 0
mask2 = 0
boxes = None
skipped_frames = -1

#Create video writer
# ================================================================================= #
if option == 1:
    video_path = "ResultVideo/raw_track.avi"
    image_path = "raw_track_65.png"
elif option == 2:
    video_path = "ResultVideo/improved_track.avi"
    image_path = "improved_track_65.png"
elif option == 3:
    video_path = "ResultVideo/improved_track_SSD.avi"
    image_path = "improved_track_SSD_65.png"
    
# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
# r_video = cv2.VideoWriter( video_path, fourcc, fps, (1280,720))
r_video = cv2.VideoWriter( video_path, cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    return image

def carROI(image, conf_threshold, padding, net, CLASSES, COLORS):
    mask1 = np.zeros(image.shape[:2], dtype=np.uint8)
    mask2 = np.zeros(image.shape[:2], dtype=np.uint8)
    detection_count = 0
    image2 = image
    image = preprocess(image)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    (h, w) = image.shape[:2]

    # begin to detect the car
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            idx = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #output = str(startX) + "," + str(startY) + "," + str(endX) + "," + str(endY) + ":"

            # if the car has been detected
            if idx == 7:
                detection_count += 1
                # define a mask ROI for feature detection
                # create a mask image filled with zeros, the size of original image

                # draw your selected ROI on the mask image
                tlx = startX - padding
                tly = startY - padding
                brx = endX + padding
                bry = endY + padding
                if tlx < 0:
                    tlx = 0
                if tly < 0:
                    tly = 0
                if brx > w - 1:
                    trx = w - 1
                if bry > h - 1:
                    bry = h - 1

                cv2.rectangle(mask1, (startX, startY), (endX, endY), (255), thickness=-1)
                cv2.rectangle(mask2, (tlx, tly), (brx, bry), (255), thickness=-1)

                # =========================== Correction ======================================== #
                # Draw bounding box for SSD detections
                cv2.rectangle(image2, (startX, startY), (endX, endY), COLORS[idx], thickness=3)
                # ------------------------------------------------------------------------------- #

    return image2, mask1, mask2, detection_count

def carsROI(image, conf_threshold, net, CLASSES, COLORS):
    detection_count = 0
    image2 = image
    image = preprocess(image)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    (h, w) = image.shape[:2]
    bboxes = []
    # begin to detect the car
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            idx = int(detections[0, 0, i, 1])
            # if the car has been detected
            if idx == 7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                #output = str(startX) + "," + str(startY) + "," + str(endX) + "," + str(endY) + ":"
                bboxes.append([])
                bboxes[detection_count].append(startX)
                bboxes[detection_count].append(startY)
                bboxes[detection_count].append(endX-startX+1)
                bboxes[detection_count].append(endY - startY + 1)
                detection_count += 1
                # define a mask ROI for feature detection
                # create a mask image filled with zeros, the size of original image

                # =========================== Correction ======================================== #
                # Draw bounding box for SSD detections
                cv2.rectangle(image2, (startX, startY), (endX, endY), COLORS[idx], thickness=3)
                # ------------------------------------------------------------------------------- #

    return image2, bboxes, detection_count


if capture.isOpened() is False:
    print("Error opening the video file!")

# ==============================Initialize tracker ==================================== #
if option == 3:
    # Skip all frames where there is no car detection
    while cars == 0:
        img_count += 1
        print ("Processing frame %d." % (img_count))
        skipped_frames += 1
        ret, frame = capture.read()
        m1 = frame
        #frame below has SSD detection boxes
        frame, boxes, cars = carsROI(frame, conf_threshold, net, CLASSES, COLORS)
    # start OpenCV object tracker using the supplied bounding box
    # coordinates, then start the FPS throughput estimator as well
   
    #For multi object tracking look out for: https://www.pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/
    
    tracker.init(m1, (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))
    r_video.write(frame)
    print (boxes[0][:])
#     cv2.waitKey(0)
    print ("Tracker initialized using SSD detected region")
    # fps = FPS().start()
else:
    # Press c on keyboard to capture object of interest
    while cv2.waitKey(20) & 0xFF != ord('c'):
        ret, frame = capture.read
        img_count += 1
        print ("Processing frame %d." % (img_count))
        #cv2.imshow("Tracked frames", frame)
        cv2.waitKey(10)

    # select the bounding box of the object we want to track (make
    # sure you press ENTER or SPACE after selecting the ROI)
    boxes[0][:] = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
    
    m1 = frame
    # start OpenCV object tracker using the supplied bounding box
    # coordinates, then start the FPS throughput estimator as well
    tracker.init(m1, boxes[0][:])
    r_video.write(frame)
    print ("Tracker initialized using user selected region.")
    # fps = FPS().start()

# get dimensions of image
dimensions = frame.shape
img_h = dimensions[0]
img_w = dimensions[1]
print (dimensions)
# --------------------------------------------------------------------------------- #

m2 = 0
f2 = 0
haveBox = True
# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    img_count += 1
    print("Processing frame %d." % img_count)
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()
    m2 = frame
    if ret is False:
        break
    
    # If features detections were made in previous frame
    if haveBox is True:

        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        boxes[0][:] = box[:]
        # check to see if the tracking was a success
        if success:
            print ("Successfully tracked.")
            (x, y, w, h) = [int(v) for v in box]
            print ("Coordinates", x, y, w, h)
            coordinates.append((x+int(w/2),y+int(h/2)))

            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            for i in range(len(coordinates)):
                frame = cv2.circle(frame, coordinates[i], radius, color, thickness)
            # update the FPS counter
            # fps.update()
            # fps.stop()
            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", "KCF tracker"),
                ("Success", "Yes" if success else "No")
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, img_h - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #cv2.imshow("Tracked frames", frame)
            cv2.waitKey(10)
            r_video.write(frame)
        else:
            print ("Tracking failed.")
            haveBox = False
    #if in the same iteration the tracker failed then give SSD a chance
    if haveBox is False:
        # frame below has SSD detection boxes
        cars = 0
        frame, boxes, cars = carsROI(frame, conf_threshold, net, CLASSES, COLORS)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
            
        #------- Project Part B --------------#
        # Which tracker needs to follow which SSD detected region
        # Multiple trackers were used for multiple objects
        # SSD would have detected multiple objects
        #   Some correctly others incorrectly
        # partial solution here: https://stackoverflow.com/questions/52245231/multi-object-tracking-using-opencv-python-delete-trackers-for-the-objects-that
            
            
        if cars != 0:
            print ("SSD has detected a vehicle.")
            # printing the list using loop 
            for value in boxes:
                print (value)
#               cv2.waitKey(0)
            haveBox = True
            tracker = cv2.TrackerKCF_create()
#           expression_if_true if condition else expression_if_false
            width = boxes[0][2] if boxes[0][0] + boxes[0][2] < img_w-1 else img_w - boxes[0][0]-1
            height = boxes[0][3] if boxes[0][1] + boxes[0][3] < img_h-1 else img_h - boxes[0][1]-1
#           tracker.init(m2, (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]))
            tracker.init(m2, (boxes[0][0], boxes[0][1], width, height))
            print ("Tracker re-initialized.")
            r_video.write(frame)
            print ("[%d, %d, %d, %d]." % (boxes[0][0], boxes[0][1], width, height))
            # fps = FPS().start()
            #cv2.imshow("Tracked frames", frame)
            cv2.waitKey(10)
        else:
            skipped_frames += 1
            #cv2.imshow("Tracked frames", m2)
            cv2.waitKey(10)
#         cv2.imshow("Tracked frames", frame)
        # Press q on keyboard to exit the program
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        # Break the loop

print ("Skipped frames are %d from a total of %d." % (skipped_frames, img_count))
# Release everything
r_video.release()
capture.release()
cv2.destroyAllWindows()

