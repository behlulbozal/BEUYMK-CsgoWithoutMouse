import cv2
import mediapipe as mp
import time
import math
import numpy as np
import autopy as ap
from autopy.mouse import*


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)

                self.lmList.append([id, cx, cy])

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []

        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 0))
            cv2.circle(img, (x2, y2), r, (255, 0, 0))
            cv2.circle(img, (cx, cy), r, (255, 255, 0))
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


#---------------------------------------------------------------

wcam, hcam = 640, 480
frameR = 100
smooth = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0
pTime = 0
cTime = 0
detector = handDetector(maxHands=1)
screenw, screenh = ap.screen.size()

#---------------------------------------------------------------

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

while True:
    success , img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList)!= 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        cv2.rectangle(img, (0, 0),
                      (((wcam-frameR)*2), ((hcam-frameR)*2)), (0, 255, 0), 2)
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wcam-frameR), (0,screenw))
            y3 = np.interp(y1, (frameR, hcam-frameR), (0, screenh))

            clocX = plocX + (x3-plocX) / smooth
            clocY = plocY + (y3-plocY) / smooth

            ap.mouse.move(screenw-clocX,clocY)
            cv2.circle(img, (x1, y1), 15, (255,0,0))

            plocX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (255, 255, 0 ), cv2.FILLED)
                ap.mouse.click(ap.mouse.Button.LEFT)
                time.sleep(0.5)

        if fingers[0] == 1 and fingers[1] == 1:
            length, img, lineInfo = detector.findDistance(4, 8, img)
            if length < 30:
                cv2.circle(img, (lineInfo[3], lineInfo[4]),
                           15, (255, 255, 0), cv2.FILLED)
                ap.mouse.click(ap.mouse.Button.RIGHT)
                time.sleep(0.5)
        
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (125, 255, 0), 3)
    

    cv2.imshow("KrakenDev", img)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
