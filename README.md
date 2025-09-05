# Coloured shapes detection
This project uses **openCV** to detect geometric shapes in images and their colour too!

## How does it work??
- **Preprocessing:**
Converting the image to grayscale. Then applying thresholding to isolate shapes. After that, extracting contours with cv2.findContours.

- **Shape Detection:**
Approximating contours with cv2.approxPolyDP, then classifying by number of vertices
**(Triangle --> 3 / Square --> 4 (checked with aspect ratio) / Circle --> from 6 to 8 / Line --> other)*

- **Colour Detection:**
Converting image to HSV colour space
Computing mean hue value inside the contour mask
Mapping hue ranges to colour names

- **Annotation:**
Drawing contours on the image
Labelling each shape with its colour
Saving the output (Here is the final output btw: 
