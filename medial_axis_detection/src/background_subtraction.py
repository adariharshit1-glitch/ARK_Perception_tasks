import cv2

def subtract_background(frame, bg_subtractor):

    mask = bg_subtractor.apply(frame)
    
    return mask