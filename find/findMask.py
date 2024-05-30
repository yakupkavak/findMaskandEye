import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd
import cProfile
import pstats


def main():
    cam = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    mounth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

    virus = cv2.imread("virus.jpg")
    tik = cv2.imread("itsokey.png")
    tik = cv2.resize(tik,(120,120))

    false = cv2.imread("falseicon.png")
    false = cv2.resize(false,(120,120))

    virus_gray = cv2.cvtColor(virus, cv2.COLOR_BGR2GRAY)
    _, masked = cv2.threshold(virus_gray, 10, 255, cv2.THRESH_BINARY)

    cv2.namedWindow("Find Mask and Eye", cv2.WINDOW_AUTOSIZE)
    camX, camY = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cerceve = np.zeros((camY, camX + 100, 3), np.uint8)
    print(cerceve.shape[1])
    cerceve[0:120, 0:120] = (255,255,255)

    cv2.putText(cerceve,"Detected",(12,14),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)
    cv2.putText(cerceve,"Eye",(32,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)

    checkEye = False

    #GÖZÜ YÜZÜN İÇERİSİNDE ARAMIYOR
    def nothing(int):
        pass


    while cam.isOpened():
        _, frame = cam.read()

        video = frame.copy()

        sumMounthX = 0
        sumMounthY = 0
        mask_wearing = False
        maskX, maskY, maskH, maskW = 0,0,0,0

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 14,minSize= (100,100))

        for (x, y, w, h) in faces:
            mask_wearing = False
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

            maskX, maskY, maskH, maskW = x, y, h, w
            eye_gray = gray_frame[y:y + h, x:x + w]  #EYE GRAY DOĞRU ÇALIŞIYOR

            eyes = eye_cascade.detectMultiScale(eye_gray, 1.25, 15)
            cerceve[40:140, :120] = (255, 255, 255)
            for (ex, ey, ew, eh) in eyes:
                eyeCount = 0
                #cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + eh + ey), (155, 0, 0),2)
                sumMounthX = ex + ew
                sumMounthY = eh + ey

                if (eyeCount == 0):
                    currentEye = video[y + ey:y + sumMounthY, x + ex:x + sumMounthX]
                    #currentEye = cv2.GaussianBlur(currentEye, (3, 3), 1)
                    eyeGray = cv2.cvtColor(currentEye, cv2.COLOR_BGR2GRAY)
                    #TO GET EYE COLOR WE NEED TO MATRIX OF EYE
                    #CONTROL THE MASK SECTİON
                    below_eye = eye_gray[int(sumMounthY * 1.2):, :]

                    if below_eye.size != 0:
                        #YAKIN MESAFE KONTROLÜ
                        cv2.imshow("kontrol",below_eye)
                        if (below_eye.shape[1] > 140):
                            #below_eye = cv2.GaussianBlur(below_eye, (5, 5), 1.2)
                            mouths = mounth_cascade.detectMultiScale(below_eye, 1.3, 8)
                            for (mx, my, mw, mh) in mouths:
                                cv2.rectangle(frame, (x + mx, my + int(sumMounthY * 1.2) + y),
                                              (x + mx + mw, my + int(sumMounthY * 1.2) + y + mh), (0, 0, 0), 2)
                                mask_wearing = True

                        #UZAK MESAFE KONTROLÜ
                        else:
                            #below_eye = cv2.GaussianBlur(below_eye, (3, 3), 1)
                            mouths = mounth_cascade.detectMultiScale(below_eye, 1.11, 6, minSize=(20, 20))
                            for (mx, my, mw, mh) in mouths:
                                cv2.rectangle(frame, (x + mx, my + int(sumMounthY * 1.2) + y),
                                             (x + mx + mw, my + int(sumMounthY * 1.2) + y + mh), (0, 0, 0), 2)
                                mask_wearing = True

                myFace = frame[maskY:maskY + maskH, maskX:maskX + maskW]

                cerceve[360:480, :120] = (255, 255, 255)
                if mask_wearing:

                    virus_resize = cv2.resize(virus, (maskW, maskH))
                    masked_resize = cv2.resize(masked, (maskW, maskH))
                    masked_inverted = cv2.bitwise_not(masked_resize)
    
                    img_bg = cv2.bitwise_and(myFace, myFace, mask=masked_inverted)
                    img_fg = cv2.bitwise_and(virus_resize, virus_resize, mask=masked_resize)
                    final_img = cv2.add(img_bg, img_fg)

                    frame[maskY:maskY + maskH, maskX:maskX + maskW] = final_img

                    cerceve[360:480, :120] = false

                else:
                    cerceve[360:480, :120] = tik
                    print("maske takılıyor")

        cerceve[:, 100:] = frame


        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord("o"):
            checkEye = True
        elif key == ord("c"):
            checkEye = False

        if checkEye:
            cv2.rectangle(cerceve,(150,50),(220,110),(0,255,0),2)
            getEyeGray = gray_frame[50:110, 150:220]
            myEye = frame[50:100,150:220]
            minDist = 20
            param1 = 52
            param2 = 50
            circles = cv2.HoughCircles(getEyeGray, cv2.HOUGH_GRADIENT, 1.2, minDist, param1=param1, param2=param2,
                                       minRadius=20, maxRadius=80)
            if circles is not None:
                circles = np.uint16(np.around(circles))

                for a, b, r in circles[0, :]:  # 0.indexten tamamını alıyoruz
                    cv2.circle(myEye, (a, b), r, (0, 255, 255), -1)

            eyeX = myEye.shape[0]
            eyeY = myEye.shape[1]
            cerceve[40:90, 20:90] = myEye

        cv2.imshow("Find Mask and Eye", cerceve)
    cv2.destroyAllWindows()





if __name__ == '__main__':
    # Profiling şlemini başlat
    profiler = cProfile.Profile()
    profiler.enable()

    # Ana fonksiyonu çalıştır
    main()

    # Profiling'i durdur
    profiler.disable()

    # Profiling sonuçlarını görüntüleme
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()