import cv2
import pandas as pd
import os
import numpy as np

# saving the output after execution
save_output = "min"
os.makedirs(save_output, exist_ok=True)

# reading & scaling the test img
photo = cv2.imread("test.jpg")
img = cv2.resize(photo, (1162, 540))

# deteting the edges + contour
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    area=cv2.contourArea(contour)
    if area < 20:
        continue

    area=cv2.contourArea(contour)
    approx = cv2.approxPolyDP(contour, 0.02* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [contour], 0, (255), 2)
    x, y , w, h = cv2.boundingRect(approx)

    aspect_ratio = float(w) / h  #width/height to avoid the confusion between lines and squares
    
    if len(approx) == 4 and aspect_ratio == 1:
        cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255), 1, cv2.LINE_AA)
        
    elif len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255), 1, cv2.LINE_AA)
        
    elif 5 < len(approx) < 9:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255), 1, cv2.LINE_AA)
        
    else:
        cv2.putText(img, "Line", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255), 1, cv2.LINE_AA)


    # color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(hsv, mask=mask)
    hue = mean_val[0]

    def get_color_name(hue):
        if 0 <= hue < 10 or 160 <= hue <= 180:
            return "Red"
        
        elif 25 <= hue < 40:  
            return "Yellow"
        
        elif 40 <= hue < 85:
            return "Green"
        
        elif 85 <= hue < 140:
            return "Blue"

    color_name = get_color_name(hue)

    # Add both shape + color label
    cv2.putText(img, f"{color_name}", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

    
    cv2.imshow("test",img)
    cv2.waitKey(1000)
    filename = "test.png"
    cv2.imwrite(os.path.join(save_output, filename), img)
