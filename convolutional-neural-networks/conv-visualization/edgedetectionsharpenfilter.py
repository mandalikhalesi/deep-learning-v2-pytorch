# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time

# read in the image
start = time.time()
print("\nReading image...")
image = mpimg.imread('./data/curved_lane.jpg')
plt.imshow(image)
end = time.time()
print("\nTime to process...{:.2f} seconds".format(end-start))

# convert image to grayscale
start = time.time()
print("\nConverting to greyscale...")
grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(grey, cmap='gray')
end = time.time()
print("\nTime to process...{:.2f} seconds".format(end-start))

# Create custom convolutional kernels Sobel Y and X and _1

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

filter_1 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])


# Define filters and display resulting images at each step
start = time.time()
print("\nDefining filters...")
filtered_img_sobely = cv2.filter2D(grey, -1, sobel_y)
filtered_img_sobelx = cv2.filter2D(filtered_img_sobely, -1, sobel_x)
filtered_img_filter_1 = cv2.filter2D(image, -1, filter_1)

print("\nImage after Sobel Y filter...\n")
plt.imshow(filtered_img_sobely, cmap='gray')
plt.title("SobelY filter")
print("\nThen after running that img through Sobel X filter...\n")
plt.imshow(filtered_img_sobelx, cmap='gray')
plt.title("SobelX filter")
plt.imshow(filtered_img_filter_1)
plt.title("_1 filter")
end = time.time()
print("\nDone. Time to process...{:.2f} seconds".format(end-start))

# %%