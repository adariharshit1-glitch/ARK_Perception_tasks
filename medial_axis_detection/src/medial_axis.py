import cv2
import numpy as np

def find_parallel_lines(lines, angle_threshold=0.1):

    best_pair = None
    max_dist = 0

    for i in range(len(lines)):
        
        for j in range(i+1, len(lines)):

            rho1, theta1 = lines[i]
            rho2, theta2 = lines[j]

            if abs(theta1 - theta2) < angle_threshold:

                dist = abs(rho1 - rho2)

                if dist > max_dist:
                    max_dist = dist
                    best_pair = (lines[i], lines[j])

    if best_pair is None:
        return None, None

    return best_pair


def compute_medial_axis(line1,line2):

    rho1,theta1 = line1
    rho2,theta2 = line2

    rho_mid = (rho1 + rho2)/2
    theta_mid = (theta1 + theta2)/2

    return rho_mid,theta_mid

def draw_line(image,rho,theta,color):

    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1500 * (b) * (-1))
    y1 = int(y0 + 1500 * (a))

    x2 = int(x0 - 1500 * (b) * (-1))
    y2 = int(y0 - 1500 * (a))

    cv2.line(image, (x1,y1), (x2,y2), color, 2)