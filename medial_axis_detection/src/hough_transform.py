import numpy as np

def hough_lines(edges):

    height, width = edges.shape

    rho_max = int(np.sqrt(height**2 + width**2))

    thetas = np.deg2rad(np.arange(-90, 90))

    accumulator = np.zeros((2*rho_max, len(thetas)), dtype=np.uint64)

    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):

        x = x_idxs[i]
        y = y_idxs[i]

        for t in range(len(thetas)):
            theta = thetas[t]
            rho = int(x*np.cos(theta) + y*np.sin(theta))
            accumulator[rho + rho_max, t] += 1

    return accumulator, thetas, rho_max

def detect_lines(accumulator, thetas, rho_max, threshold = 97):

    lines = []

    for r in range(accumulator.shape[0]):
        for t in range(accumulator.shape[1]):
            if accumulator[r, t] > threshold:
                rho = r - rho_max
                theta = thetas[t]
                lines.append((rho, theta))

    return lines

