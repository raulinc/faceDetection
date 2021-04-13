import dlib
import cv2
detector = dlib.get_frontal_face_detector()
image= cv2.imread('test1.jpg')
img=cv2.resize(image,(800,600))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray, 1) # result
#to draw faces on image
for result in faces:
    x = result.left()
    y = result.top()
    x1 = result.right()
    y1 = result.bottom()
    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

cv2.imshow('img',img,)
cv2.waitKey()