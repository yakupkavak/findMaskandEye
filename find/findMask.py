import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mounth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

virus = cv2.imread("virus.jpg")
virus_gray = cv2.cvtColor(virus, cv2.COLOR_BGR2GRAY)
_, masked = cv2.threshold(virus_gray, 10, 255, cv2.THRESH_BINARY)

cv2.namedWindow("Find Mask and Eye", cv2.WINDOW_AUTOSIZE)
camX, camY = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
cerceve = np.zeros((camY, camX + 100, 3), np.uint8)
cerceve[0:100, 0:100] = (255,255,255)
cv2.putText(cerceve,"Detected",(12,14),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)
cv2.putText(cerceve,"Eye",(32,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)


#GÖZÜ YÜZÜN İÇERİSİNDE ARAMIYOR
def nothing(int):
    pass


while cam.isOpened():
    _, frame = cam.read()

    video = frame.copy()
    cerceve[:,100:] = video
    sumMounthX = 0
    sumMounthY = 0
    mask_wearing = False
    maskX = 0
    maskY = 0
    maskH = 0
    maskW = 0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = frame.copy()

    faces = face_cascade.detectMultiScale(gray_frame, 1.25, 10)

    for (x, y, w, h) in faces:
        mask_wearing = False
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
        maskX, maskY, maskH, maskW = x, y, h, w
        eye_gray = gray_frame[y:y + h, x:x + w]  #EYE GRAY DOĞRU ÇALIŞIYOR

        eyes = eye_cascade.detectMultiScale(eye_gray, 1.25, 15)

        for (ex, ey, ew, eh) in eyes:
            eyeCount = 0
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + eh + ey), (155, 0, 0),
                          2)  ##ALGILANAN ÇEVRE SORUNLU
            check = frame[y + ey:y + eh + ey, x + ex:x + ex + ew]
            sumMounthX = ex + ew
            sumMounthY = eh + ey

            if (eyeCount == 0):
                # Get the eye Color
                currentEye = frame[y + ey:y + sumMounthY, x + ex:x + sumMounthX]
                currentEye = cv2.GaussianBlur(currentEye, (1, 1), 1)
                eyeGray = cv2.cvtColor(currentEye, cv2.COLOR_BGR2GRAY)

                currentEye = cv2.resize(currentEye, (80, 80))
                eyeGray = cv2.resize(eyeGray, (80, 80))
                currentEye = currentEye[15:68, 15:68]
                eyeGray = eyeGray[15:68, 15:68]

                cv2.imshow("myeye", currentEye)

                minDist = currentEye.shape[0] / 8
                param1 = max(52, int(minDist))
                param2 = max(30, int(minDist / 2))
                circles = cv2.HoughCircles(eyeGray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                           minRadius=15, maxRadius=40)
                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for a, b, r in circles[0, :]:  #0.indexten tamamını alıyoruz
                        cv2.circle(currentEye, (a, b), r, (0, 255, 255), -1)

                eyeX = currentEye.shape[0]
                eyeY = currentEye.shape[1]
                cerceve[40:eyeY+40, 20:eyeX+20] = currentEye

                below_eye = eye_gray[int(sumMounthY * 1.2):, :]

                if below_eye.size != 0:
                    if (below_eye.shape[1] > 140):
                        below_eye = cv2.GaussianBlur(below_eye, (5, 5), 1.2)
                        mouths = mounth_cascade.detectMultiScale(below_eye, 1.1, 8, minSize=(40, 40))
                        for (mx, my, mw, mh) in mouths:
                            cv2.rectangle(frame, (x + mx, my + int(sumMounthY * 1.2) + y),
                                          (x + mx + mw, my + int(sumMounthY * 1.2) + y + mh), (0, 0, 0), 2)
                            mask_wearing = True
                    else:
                        below_eye = cv2.GaussianBlur(below_eye, (3, 3), 1)
                        mouths = mounth_cascade.detectMultiScale(below_eye, 1.01, 6, minSize=(20, 20))
                        cv2.imshow("month", below_eye)
                        for (mx, my, mw, mh) in mouths:
                            cv2.rectangle(frame, (x + mx, my + int(sumMounthY * 1.2) + y),
                                          (x + mx + mw, my + int(sumMounthY * 1.2) + y + mh), (0, 0, 0), 2)
                            mask_wearing = True
            eyeCount = 1
            myFace = frame[maskY:maskY + maskH, maskX:maskX + maskW]
            if mask_wearing:
                """
                virus_resize = cv2.resize(virus, (maskW, maskH))
                masked_resize = cv2.resize(masked, (maskW, maskH))
                masked_inverted = cv2.bitwise_not(masked_resize)

                img_bg = cv2.bitwise_and(myFace, myFace, mask=masked_inverted)
                img_fg = cv2.bitwise_and(virus_resize, virus_resize, mask=masked_resize)
                final_img = cv2.add(img_bg, img_fg)

                frame[maskY:maskY + maskH, maskX:maskX + maskW] = final_img

                video[maskY:maskY + maskH, maskX:maskX + maskW] = final_img
        
                #cv2.bitwise_and(newMasked,newMasked,frame)
                """
                cv2.imshow("Find Mask and Eye", cerceve)
            else:
                print("maske takılıyor")

    key = cv2.waitKey(5)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
