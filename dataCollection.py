import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 indicates the default camera (usually your webcam)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a HandDetector object
detector = HandDetector(maxHands=1)
offset = 20  # for proper croping of image
imgsize = 300

folder = "Data/f"
counter = 0

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    hands, img = detector.findHands(img)  # it will detect the handss

    # crop the original image
    if hands:
        hand = hands[0]  # bcz singlehand
        x, y, w, h = hand["bbox"]  # w-width,h-height,x-y coordinate,bbox=boundingbox
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            # all images should be same size, creating white background
            imgWhite = np.ones((imgsize, imgsize, 3),
                               np.uint8) * 255  # bcz color image(0-255, ie 8bit),3->3 channels RGB

            imgCrop = img[y - offset:y + h + offset,
                      x - offset:x + w + offset]  # starting height and starting width , which will give the exact bbox
            imgCropShape = imgCrop.shape  # contains height, width and 3 channels

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgsize / h  # k is const
                wCalc = math.ceil(k * w)  # round it to higher
                if wCalc > 0:
                    imgResize = cv2.resize(imgCrop, (wCalc, imgsize))
                else:
                    continue
                imgResizeShape = imgResize.shape
                # to shift the img to the centre
                wGap = math.ceil((imgsize - wCalc) / 2)  # this is the gap to push forward to make it centre

                # put imgcrop matrix in the imgwhitematrix
                imgWhite[:, wGap:wCalc + wGap] = imgResize  # no need of channels
            else:
                k = imgsize / w  # k is const
                hCalc = math.ceil(k * h)  # round it to higher
                if hCalc > 0:
                    imgResize = cv2.resize(imgCrop, (imgsize, hCalc))
                else:
                    continue
                imgResizeShape = imgResize.shape
                # to shift the img to the centre
                hGap = math.ceil((imgsize - hCalc) / 2)  # this is the gap to push forward to make it centre

                # put imgcrop matrix in the imgwhitematrix
                imgWhite[hGap:hCalc + hGap, :] = imgResize  # no need of channels

                # if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):  # it will save when we press 's'
        counter += 1

        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)


