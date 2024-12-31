import cv2
import numpy as np

# Load the input image
image_path = 'download.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply edge detection (Canny)
edges = cv2.Canny(image, 50, 150)

# Function to perform Hough Line Detection for specific angle ranges
def detect_lines(image, angle_range):
    lines = cv2.HoughLines(image, 1, np.pi / 180, threshold=100)
    lines_img = np.zeros_like(image)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)  # Convert from radians to degrees
            # Check if the angle is in the desired range
            if angle_range[0] <= angle <= angle_range[1] or \
               (angle_range[1] < angle_range[0] and (angle >= angle_range[0] or angle <= angle_range[1])):
                # Convert from polar coordinates to Cartesian
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(lines_img, (x1, y1), (x2, y2), 255, 1)  # Draw the line in white
    return lines_img

# Function to perform corner detection (Point Detection)
def detect_corners(image):
    corners = cv2.cornerHarris(image, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)  # Dilate the corners to make them more visible
    corner_img = np.zeros_like(image)
    corner_img[corners > 0.01 * corners.max()] = 255  # Mark corners with white
    return corner_img

# Apply line detection with specific ranges for the angle
vertical_lines = detect_lines(edges, (-10, 10))
horizontal_lines = detect_lines(edges, (80, 100))
neg_45_lines = detect_lines(edges, (-50, -40))
pos_45_lines = detect_lines(edges, (40, 50))

# Apply point detection (corner detection)
corners = detect_corners(image)

# Edge detection (Sobel) for specific orientations
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Edge detection in the x direction (vertical)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Edge detection in the y direction (horizontal)
sobel_45_pos = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)  # Approximate edge detection in +45°
sobel_45_neg = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3) - cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Approximate -45°

# Convert Sobel results to absolute values and then to uint8 format
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_45_pos = cv2.convertScaleAbs(sobel_45_pos)
sobel_45_neg = cv2.convertScaleAbs(sobel_45_neg)

# Organize images in a 3x3 grid using OpenCV hconcat and vconcat
top_row = cv2.hconcat([image, vertical_lines, horizontal_lines])
middle_row = cv2.hconcat([sobel_45_neg, sobel_45_pos, sobel_x])
bottom_row = cv2.hconcat([sobel_y, corners, np.zeros_like(image)])

# Stack the rows vertically to create the final grid image
final_image = cv2.vconcat([top_row, middle_row, bottom_row])

# Display the final image
cv2.imshow("Line, Edge, and Point Detection", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
