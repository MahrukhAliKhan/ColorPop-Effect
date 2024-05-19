#import the libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#read the image
img = cv.imread("path to your testing image")
#convert the BGR image to HSV colour space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#obtain the grayscale image of the original image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#set the bounds for the red hue
lower_red = np.array([160,100,50])
upper_red = np.array([180,255,255])

#create a mask using the bounds set
mask = cv.inRange(hsv, lower_red, upper_red)
#create an inverse of the mask
mask_inv = cv.bitwise_not(mask)
#Filter only the red colour from the original image using the mask(foreground)
res = cv.bitwise_and(img, img, mask=mask)
#Filter the regions containing colours other than red from the grayscale image(background)
background = cv.bitwise_and(gray, gray, mask = mask_inv)
#convert the one channelled grayscale background to a three channelled image
background = np.stack((background,)*3, axis=-1)
#add the foreground and the background
added_img = cv.add(res, background)

# Save the images to files or display them using matplotlib
cv.imwrite("back.jpg", background)
cv.imwrite("mask_inv.jpg", mask_inv)
cv.imwrite("added.jpg", added_img)
cv.imwrite("mask.jpg", mask)
cv.imwrite("gray.jpg", gray)
cv.imwrite("hsv.jpg", hsv)
cv.imwrite("res.jpg", res)

# Display the images using matplotlib if needed
plt.imshow(cv.cvtColor(background, cv.COLOR_BGR2RGB))
plt.title('Background')
plt.axis('off')
plt.show()

plt.imshow(mask_inv, cmap='gray')
plt.title('Mask Inverted')
plt.axis('off')
plt.show()

plt.imshow(cv.cvtColor(added_img, cv.COLOR_BGR2RGB))
plt.title('Added Image')
plt.axis('off')
plt.show()

plt.imshow(mask, cmap='gray')
plt.title('Mask')
plt.axis('off')
plt.show()

plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

plt.imshow(cv.cvtColor(hsv, cv.COLOR_HSV2RGB))
plt.title('HSV Image')
plt.axis('off')
plt.show()

#if cv.waitKey(0):
# cv.destroyAllWindows()
