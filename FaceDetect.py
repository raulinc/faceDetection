import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h = face

            cut = frame[y:y+h,x:x+w]
            fix = cv2.resize(cut, (200, 200))
            gray  = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

        #cv2.imshow("MY face",gray)
        cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()