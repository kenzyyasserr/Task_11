import cv2
import pandas as pd
import os

# saving the output after execution
save_output = "min"
os.makedirs(save_output, exist_ok=True)

# reading the test img
photo=cv2.imread("test.jpg")
img=cv2.resize(photo,None,fx=900/photo.shape[1],fy=900/photo.shape[0])