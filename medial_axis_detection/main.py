import cv2

from src.extract_frames import extract_frames
from src.background_subtraction import subtract_background
from src.image_cleaning import clean_image
from src.edge_detection import detect_edges
from src.hough_transform import hough_lines, detect_lines
from src.medial_axis import find_parallel_lines, compute_medial_axis, draw_line


video_path = "data/videos/1.mp4"
#video_path = "data/videos/2.mp4"
#video_path = "data/videos/3.mp4"

frames = extract_frames(video_path)
frame_index = 0
target_frame = 3

bg_subtractor = cv2.createBackgroundSubtractorMOG2()

cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("clean", cv2.WINDOW_NORMAL)
cv2.namedWindow("edges", cv2.WINDOW_NORMAL)
cv2.namedWindow("medial axis", cv2.WINDOW_NORMAL)

for frame in frames:

    frame_index += 1

    if frame_index == target_frame:

        cv2.imwrite("results/frames/1_input.jpg", frame)

    mask = subtract_background(frame, bg_subtractor)

    clean = clean_image(mask)

    edges = detect_edges(clean)

    accumulator, thetas, rho_max = hough_lines(edges)

    lines = detect_lines(accumulator, thetas, rho_max)

    line1, line2 = find_parallel_lines(lines)

    if line1 is not None:

        rho_mid, theta_mid = compute_medial_axis(line1, line2)

        draw_line(frame, rho_mid, theta_mid, (0,255,0))

    if frame_index == target_frame:

        cv2.imwrite("results/frames/1_mask.jpg", mask)
        cv2.imwrite("results/frames/1_clean.jpg", clean)
        cv2.imwrite("results/frames/1_edges.jpg", edges)
        cv2.imwrite("results/frames/1_final.jpg", frame)

    cv2.imshow("mask", mask)
    cv2.imshow("clean", clean)
    cv2.imshow("edges", edges)
    cv2.imshow("medial axis", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break


cv2.destroyAllWindows()