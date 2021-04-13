import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

data = np.load("face_data.npy")

print(data.shape, data.dtype)

X = data[:, 1:].astype(np.uint8)
Y = data[:, 0]

model = KNeighborsClassifier()
model.fit(X, Y)

cap = cv2.VideoCapture(0 , cv2.CAP_DSHOW)

while True:
    ret , frame = cap.read()
    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x , y, w, h = face

            cut = frame[y:y+h , x:x+w]

            fix = cv2.resize(cut, (200,200))
            gray = cv2.cvtColor(fix,cv2.COLOR_BGR2GRAY)

            out = model.predict([gray.flatten()])

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame, str(out[0]),(x, y - 10),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0))

            print(out)

        cv2.imshow("Frame",frame)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()