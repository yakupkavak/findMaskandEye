import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint as rnd
import cProfile
import pstats


def main():
    cam = cv2.VideoCapture(0)
    iris = False
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    mounth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

    virus = cv2.imread("virus.jpg")

    tik = cv2.imread("tickicon.png",1)
    tik = cv2.resize(tik,(120,120))

    false = cv2.imread("delete.png",1)
    false = cv2.resize(false,(120,120))


    virus_gray = cv2.cvtColor(virus, cv2.COLOR_BGR2GRAY)
    _, masked = cv2.threshold(virus_gray, 10, 255, cv2.THRESH_BINARY)

    cv2.namedWindow("Find Mask and Eye", cv2.WINDOW_AUTOSIZE)
    camX, camY = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cerceve = np.zeros((camY, camX + 100, 3), np.uint8)
    cerceve[121:360,:120] = (255,255,255)
    cerceve[0:119, 0:120] = (255,255,255)

    eye_color_matrix = np.zeros((60, 120, 3), dtype=np.uint8)

    cv2.putText(cerceve,"Detect",(12,14),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)
    cv2.putText(cerceve,"Eye",(22,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)
    cv2.putText(cerceve,"Detect",(12,134),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)
    cv2.putText(cerceve,"Iris",(22,148),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,0,255),1)
    cv2.putText(cerceve, "Detected", (12, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 255), 1)
    cv2.putText(cerceve, "Color", (22, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 0, 255), 1)

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

            for (ex, ey, ew, eh) in eyes:
                eyeCount = 0
                sumMounthX = ex + ew
                sumMounthY = eh + ey

                if (eyeCount == 0):
                    currentEye = video[y + ey:y + sumMounthY, x + ex:x + sumMounthX]
                    #TO GET EYE COLOR WE NEED TO MATRIX OF EYE
                    #CONTROL THE MASK SECTİON
                    below_eye = eye_gray[int(sumMounthY * 1.2):, :]

                    if below_eye.size != 0:
                        #YAKIN MESAFE KONTROLÜ
                        if (below_eye.shape[1] > 140):
                            mouths = mounth_cascade.detectMultiScale(below_eye, 1.3, 8)
                            for (mx, my, mw, mh) in mouths:
                                mask_wearing = True

                        #UZAK MESAFE KONTROLÜ
                        else:
                            mouths = mounth_cascade.detectMultiScale(below_eye, 1.01, 6, minSize=(20, 20))
                            #check to identfiy mouth
                            for (mx, my, mw, mh) in mouths:
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
                    checkEye = False

                else:
                    cerceve[360:480, :120] = tik
                    print("maske takılıyor")

        cerceve[:, 100:] = frame

        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord("o"):

            if mask_wearing is False:
                checkEye = True

        elif key == ord("c"):
            checkEye = False
            cerceve[40:110, :90] = (255, 255, 255)

        if checkEye:
            if iris:
                cv2.rectangle(cerceve, (250, 100), (320, 170), (0, 255, 0), 2)
            else:
                cv2.rectangle(cerceve, (250, 100), (320, 170), (0, 0, 255), 2)
            cerceve[40:110, :90] = (255, 255, 255)
            cerceve[278:360, :90] = (255, 255, 255)
            getEyeGray = gray_frame[100:170, 150:220]
            myEye = frame[100:170, 150:220]
            minDist = 30
            param1 = 54
            param2 = 22

            circles = cv2.HoughCircles(getEyeGray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2,
                                       minRadius=14, maxRadius=18)

            if circles is not None:
                circles = np.uint16(np.around(circles))

                for a, b, r in circles[0, :]:  # 0.indexten tamamını alıyoruz

                    print("eye algılandı")

                    cv2.circle(getEyeGray, (a, b), r, (0,0,0), -1)
                    _, maskedEye = cv2.threshold(getEyeGray, 10, 255, cv2.THRESH_BINARY)

                    reverseMaskedEye = cv2.bitwise_not(maskedEye)

                    eye_bg = cv2.bitwise_and(myEye,myEye,mask= maskedEye)
                    eye_fg = cv2.bitwise_and(myEye, myEye, mask=reverseMaskedEye)

                    eye_fg_gray = cv2.cvtColor(eye_fg, cv2.COLOR_BGR2GRAY)

                    newCircle = cv2.HoughCircles(eye_fg_gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=15, param2=11,
                                               minRadius=3, maxRadius=6)

                    #göz bebeği çıkarılıyor eye_fg den
                    if newCircle is not None:
                        print("gözbebeği algılandı")
                        newCircle = np.uint16(np.around(newCircle))

                        for ca, cb, cr in newCircle[0, :]:
                            #GÖZ BEBEĞİ ALGILANDI
                            iris = True
                            cv2.circle(eye_fg_gray, (ca, cb), cr, (0, 0, 0), -1)
                            _, cMaskedEye = cv2.threshold(eye_fg_gray, 10, 255, cv2.THRESH_BINARY)

                            eye_new_fg = cv2.bitwise_and(myEye, myEye, mask=cMaskedEye)

                            irisShape = eye_new_fg.shape[0]
                            iris_mean_color2 = cv2.mean(eye_new_fg, mask=cMaskedEye)[:3]

                            cerceve[40:110, 20:90] = myEye

                            cerceve[160:160+irisShape,20:20+irisShape] = eye_new_fg
                            b,g,r = iris_mean_color2[0],iris_mean_color2[1],iris_mean_color2[2]

                            eyeColor = (b,g,r)
                            eye_color_matrix[:,:] = eyeColor
                            cerceve[300:360,:120] = eyeColor

                            #identify eye color
                            if r > 50 and g > 30 and b < 100:
                                cv2.putText(cerceve, "Brown", (22, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 1)
                            elif r < 150 and 100 < g < 200 and 50 < b < 150:
                                cv2.putText(cerceve, "Green", (22, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 1)
                            elif r < 130 and g < 180 and 130 < b:
                                cv2.putText(cerceve, "Blue", (22, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (b, g, r), 1)


                            checkEye = False




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