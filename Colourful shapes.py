import cv2
import numpy as np

# Read the image
image = cv2.imread(r'C:\Users\Shri\train0\img1.png')

# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for the colors you want to detect (in HSV)
lower_bound = np.array([120,100,100])  # Lower bound for blue color
upper_bound = np.array([120,100,100])  # Upper bound for blue color

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Find contours in the thresholded image
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours
for contour in contours:
    # Approximate the contour to a polygon
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Filter out small or non-convex shapes
    if cv2.contourArea(approx) > 100 and cv2.isContourConvex(approx):
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(approx)
        
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Print the coordinates
        print("Bounding box coordinates:", (x, y), (x+w, y+h))

# Display the image with bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
