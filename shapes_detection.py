import cv2
import pandas as pd
import os

# saving the output after execution
save_output = "min"
os.makedirs(save_output, exist_ok=True)

# reading the test img
photo=cv2.imread("test.jpg")
img=cv2.resize(photo,None,fx=900/photo.shape[1],fy=900/photo.shape[0])

# deteting the edges + contour
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 1)
edges = cv2.Canny(blur, threshold1=50, threshold2=150)
contours , hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    area=cv2.contourArea(contour)
    approx = cv2.approxPolyDP(contour, 0.028* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [contour], 0, (255), 2)
    x, y , w, h = cv2.boundingRect(approx)
    
    if len(approx) == 4 :
        cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 1, cv2.LINE_AA)
        
    elif len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 1, cv2.LINE_AA)
        
    elif len(approx) == 1:
        cv2.putText(img, "Line", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 1, cv2.LINE_AA)
        
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 1, cv2.LINE_AA)

    cv2.imshow("test",img)
    cv2.waitKey(1000)
    filename = "test.png"
    cv2.imwrite(os.path.join(save_output, filename), img)
