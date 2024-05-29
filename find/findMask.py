import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mounth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
cv2.namedWindow("Eyes", cv2.WINDOW_AUTOSIZE)

#GÖZÜ YÜZÜN İÇERİSİNDE ARAMIYOR
def nothing(int):
    pass

while cam.isOpened():
    _,frame = cam.read()
    sumMounthX = 0
    sumMounthY = 0
    mask_wearing = False

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    img = frame.copy()

    faces = face_cascade.detectMultiScale(gray_frame,1.25,10)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),3)

        eye_gray = gray_frame[y:y+h, x:x+w] #EYE GRAY DOĞRU ÇALIŞIYOR

        eyes = eye_cascade.detectMultiScale(eye_gray,1.25,15)

        for (ex,ey,ew,eh) in eyes:
            eyeCount = 0
            cv2.rectangle(frame,(x+ex,y+ey), (x+ex+ew,y+eh+ey),(155,0,0),2) ##ALGILANAN ÇEVRE SORUNLU
            check = frame[y+ey:y+eh+ey,x+ex:x+ex+ew]
            sumMounthX = ex+ew
            sumMounthY = eh+ey

            if(eyeCount == 0): ##BURASI PATLATIYOR BURADA FRAME DEĞŞİYOR DEMEK Kİ
                # Get the eye Color
                currentEye = frame[y+ey:y+sumMounthY, x+ex:x+sumMounthX]
                currentEye = cv2.GaussianBlur(currentEye, (1, 1), 1)
                eyeGray = cv2.cvtColor(currentEye, cv2.COLOR_BGR2GRAY)

                currentEye = cv2.resize(currentEye, (80, 80))
                eyeGray = cv2.resize(eyeGray, (80, 80))
                currentEye = currentEye[15:68,15:68]
                eyeGray = eyeGray[15:68,15:68]
                cv2.imshow("myeye",currentEye)



                minDist = currentEye.shape[0] / 8
                param1 = max(52,int(minDist))
                param2 = max(30,int(minDist/2))
                circles = cv2.HoughCircles(eyeGray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                           minRadius=15, maxRadius=40)

                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for a, b, r in circles[0, :]: #0.indexten tamamını alıyoruz
                        cv2.circle(currentEye, (a,b), r, (0, 255, 255), -1)

                eyeX= currentEye.shape[0]
                eyeY = currentEye.shape[1]
                frame[:eyeY,:eyeX] = currentEye
        eyeCount = 1

        below_eye = eye_gray[int(sumMounthY*1.3):,:]

        if below_eye.size != 0:
            below_eye = cv2.GaussianBlur(below_eye, (5, 5), 1.2)
            mouths = mounth_cascade.detectMultiScale(below_eye, 1.2, 16, minSize=(30, 30))
            for (mx, my, mw, mh) in mouths:
                cv2.rectangle(frame, (x + mx, my + int(sumMounthY * 1.3) + y),
                              (x + mx + mw, my + int(sumMounthY * 1.3) + y + mh), (0, 0, 0), 2)
                mask_wearing = True

    cv2.imshow("img",frame)

    key = cv2.waitKey(5)
    if key == ord("q"):
        break

cv2.destroyAllWindows()

