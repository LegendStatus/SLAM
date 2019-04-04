import cv2, os

vc = cv2.VideoCapture("0027.avi")
c = 1
if vc.isOpened():
	rval, frame = vc.read()
else:
	rval = False
while rval:
	rval, frame = vc.read()
	cv2.imwrite(os.getcwd()+'\\Video27\\'+str(c)+'.jpg',frame)
	c = c + 1
	cv2.waitKey(1)
vc.release()
