import cv2 as cv
import HandTracking as ht
capture = cv.VideoCapture(0)
hnd = ht.Hand(min_detection_confidence=0.9)

while True:
    isframe , img = capture.read()
    hnd.detect_hands(img)
    lmfinguer = hnd.give_hand_point(img , draw=False)
    if (len(lmfinguer)!=0):
        counting = []
        for i in range(6,20,4):
            if (lmfinguer[i+1][2] < lmfinguer[i][2]):
                counting.append(1)
            else:
                counting.append(0)
        if (lmfinguer[4][1] > lmfinguer[3][1]):
            counting.append(1)
        cv.putText(img , f'{sum(counting)}',(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(255, 0, 250) , 2,cv.LINE_AA)
    cv.imshow("Cam" , img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()
