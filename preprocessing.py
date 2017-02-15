import cv2
import numpy as np

I = cv2.imread('18.jpg')
#I2 = I

I = cv2.resize(I, None, fx = 0.5, fy = 0.5)
channels = cv2.split(I)

cv2.normalize(I,I,0,255,cv2.NORM_MINMAX)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

for i in range(0, len(channels)):
    channels[i] = clahe.apply(channels[i])

I = cv2.merge(channels)

'''
I = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
I2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)

thres = cv2.threshold(I2, 45, 255, cv2.THRESH_TRUNC)
thres = thres[1]

th = cv2.adaptiveThreshold(thres, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thI = cv2.bitwise_not(th)

I_x = cv2.addWeighted(I, 1, thI, 0.09, 0)
I_f = cv2.GaussianBlur(I_x,(3,3),0)

I = I_f

cv2.imwrite('th.jpg', thI)
'''

cv2.imwrite('I.jpg', I)
cv2.waitKey()
