import cv2

def detect_edges(image):

    edges = cv2.Canny(image, 50, 150, apertureSize = 3, L2gradient = True)
   
    return edges