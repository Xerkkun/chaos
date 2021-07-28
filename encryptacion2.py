import cv2
import numpy as np

image1 = cv2.imread("im1.jpg")
encryp = cv2.imread("encryp.png")
print(encryp.shape)
xor1 = cv2.bitwise_xor(image1, encryp)
cv2.imwrite('Xor1.png',xor1)
xor2 = cv2.bitwise_xor(xor1, encryp)
cv2.imwrite('Xor2.png',xor2)
