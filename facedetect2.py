import cv2

detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv2.imread('test1.jpg')
image = cv2.resize(img, (300, 300))
grey= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces= detector.detectMultiScale(grey,1.1,2)
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

cv2.imshow('img',image,)
cv2.waitKey()