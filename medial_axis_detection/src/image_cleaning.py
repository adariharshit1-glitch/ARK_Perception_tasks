import cv2
import numpy as np


def CVT_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)


def morphology_open(image, kernel_size):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

    return opened


def morphology_close(image, kernel_size):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    return closed


def clean_image(mask):

    median = median_filter(mask, 3)

    opened = morphology_open(median, 3)

    cleaned = morphology_close(opened, 7)

    return cleaned