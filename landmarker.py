import cv2
import dlib


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = dlib.get_frontal_face_detector()
while True:
    ret, frame = cap.read()

    faces = detector(frame)

    print(faces)
    if ret:
       key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()