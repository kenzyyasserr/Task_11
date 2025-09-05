import cv2
import numpy as np
import os

save_output = "min"
os.makedirs(save_output, exist_ok=True)

img = cv2.imread("test.jpg")
if img is None:
    raise FileNotFoundError("Couldn't read 'test.jpg' — check path/filename.")

# resize to width=900 while preserving aspect ratio
scale = 900.0 / img.shape[1]
img = cv2.resize(img, (900, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

# preprocess
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)
edges = cv2.Canny(blur, 50, 150)

# find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h_img, w_img = img.shape[:2]
min_area = max(100, int(0.0005 * h_img * w_img))  # tune this if needed

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < min_area:
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    # rotated bounding box (handles diagonal lines correctly)
    rect = cv2.minAreaRect(cnt)            # ((cx,cy),(width,height),angle)
    (cx, cy), (rw, rh), angle = rect
    rw = max(rw, 1e-6); rh = max(rh, 1e-6)
    length = max(rw, rh)
    thickness = min(rw, rh)
    thinness = thickness / length         # small -> thin object (likely a line)

    label = None

    # 1) detect lines first by rotated-box thinness + minimum length
    if thinness < 0.12 and length > 25:   # thresholds you can tune
        label = "Line"

    # 2) polygon-based classification if not already a line
    if label is None:
        if len(approx) == 3:
            label = "Triangle"
        elif len(approx) == 4:
            # use rotated box aspect ratio to decide square vs rectangle
            aspect = rw / float(rh) if rh != 0 else 0
            label = "Square" if 0.90 <= aspect <= 1.10 else "Rectangle"
        elif len(approx) > 4:
            # circularity to separate circle vs ellipse/irregular
            circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
            label = "Circle" if circularity > 0.75 else "Ellipse"
        else:
            label = "Unknown"

    # draw contour and rotated rectangle (for debugging/visualization)
    cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(img, [box], 0, (255, 0, 0), 2)

    # put text safely (keep text inside image)
    x, y, bw, bh = cv2.boundingRect(approx)
    tx = max(0, x)
    ty = max(20, y - 8)   # avoid negative y so text is visible
    cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

# show & save
cv2.imshow("labeled", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(save_output, "test_labeled.png"), img)
