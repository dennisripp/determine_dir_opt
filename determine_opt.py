import numpy as np
import cv2
import math
from enum import Enum
from typing import List, Tuple, Optional
import random
import time
import os

ym_per_pix: float = 1 / 2250
xm_per_pix: float = 1 / 2250
ranges: np.ndarray = np.array([160, 320, 480, 640])
reference_list: List[str] = []


def colortogreyImage(original_frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)


def birdseyeView(image_gray: np.ndarray) -> np.ndarray:
    width, height = image_gray.shape[1], image_gray.shape[0]
    source = np.float32([[(20, 440), (100, 180), (590, 180), (640, 470)]])
    destination = np.float32([[0, 480], [0, 0], [640, 0], [640, 480]])

    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image_gray, matrix, (width, height), flags=cv2.INTER_LINEAR)


def detectlaneLines(warped: np.ndarray) -> np.ndarray:
    threshold_low, threshold_high = 100, 200
    image_blurred = cv2.GaussianBlur(np.copy(warped), (7, 7), 0)
    return cv2.Canny(image_blurred, threshold_low, threshold_high)


def dilatelaneEdges(image_canny: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(np.copy(image_canny), kernel, iterations=1)


def polyfit(x: List[float], y: List[float], degree: int) -> List[float]:
    n = len(x)
    
    # Create the Vandermonde matrix
    X = [[x[i] ** j for j in range(degree + 1)] for i in range(n)]
    
    # Create the Y matrix
    Y = [[y[i]] for i in range(n)]
    
    # Solve the linear system using Gauss-Jordan elimination
    for i in range(degree + 1):
        for j in range(i + 1, degree + 1):
            ratio = X[j][i] / X[i][i]
            for k in range(degree + 1):
                X[j][k] -= ratio * X[i][k]
            Y[j][0] -= ratio * Y[i][0]
    
    # Back substitution
    coefficients = [0] * (degree + 1)
    for i in range(degree, -1, -1):
        for j in range(i + 1, degree + 1):
            Y[i][0] -= X[i][j] * coefficients[j]
        coefficients[i] = Y[i][0] / X[i][i]
    
    return coefficients

def detect_lane_pixels(dilated_edges: np.ndarray, warped: np.ndarray) -> Tuple[
    List[Tuple[int, int]], List[Tuple[int, int]]]:
    binary_image = np.copy(dilated_edges)
    nwindows: int = 20
    window_length: int = 70
    window_height: np.int8 = np.int8(binary_image.shape[0] // nwindows)

    bottom_fourth: np.ndarray = binary_image[binary_image.shape[0] * 7 // 8:, :]
    histogram: np.ndarray = np.sum(bottom_fourth, axis=0)

    nonzero: tuple = binary_image.nonzero()
    nonzeroy: np.ndarray = np.array(nonzero[0])
    nonzerox: np.ndarray = np.array(nonzero[1])
    midpoint: int = histogram.shape[0] // 2

    left_range_start: int = 0
    left_range_end: int = midpoint - 0
    right_range_start: int = midpoint + 0
    right_range_end: int = histogram.shape[0]

    left_histogram: np.ndarray = histogram[left_range_start:left_range_end]
    right_histogram: np.ndarray = histogram[right_range_start:right_range_end]

    threshold_histogramm: int = 500
    leftx_peak_mask: np.ndarray = left_histogram > threshold_histogramm
    rightx_peak_mask: np.ndarray = right_histogram > threshold_histogramm
    leftx_current = np.argmax(left_histogram * leftx_peak_mask) if np.any(leftx_peak_mask) else None
    rightx_current = np.argmax(right_histogram * rightx_peak_mask) + midpoint if np.any(rightx_peak_mask) else None

    left_lane_window_coord_list: List[Tuple[int, int]] = []
    right_lane_window_coord_list: List[Tuple[int, int]] = []

    for window in range(nwindows):
        win_y_low: np.int32 = dilated_edges.shape[0] - (window + 1) * window_height
        win_y_high: np.int32 = dilated_edges.shape[0] - window * window_height

        if leftx_current is not None:
            win_xleft_low = leftx_current - window_length
            win_xleft_high = leftx_current + window_length

            left_mask = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)

            if left_mask.any():
                leftx_current = np.int32(np.mean(nonzerox[left_mask]))
                lefty_current = np.int32(np.mean(nonzeroy[left_mask]))
                left_lane_window_coord_list.append((leftx_current, lefty_current))

        if rightx_current is not None:
            win_xright_low = rightx_current - window_length
            win_xright_high = rightx_current + window_length

            right_mask = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)

            if right_mask.any():
                rightx_current = np.int32(np.mean(nonzerox[right_mask]))
                righty_current = np.int32(np.mean(nonzeroy[right_mask]))
                right_lane_window_coord_list.append((rightx_current, righty_current))

    return left_lane_window_coord_list, right_lane_window_coord_list


def polynomalFitThree(left_lane_window_coord_list: List[Tuple[float, float]],
                      right_lane_window_coord_list: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    left_fit, right_fit = None, None
    window_number: int = 12

    if left_lane_window_coord_list and len(left_lane_window_coord_list) > 1:
        leftlane_x = [x for x, _ in left_lane_window_coord_list]
        leftlane_y = [y for _, y in left_lane_window_coord_list]

        if len(leftlane_x) >= window_number and len(leftlane_y) >= window_number:
            left_fit = polyfit(leftlane_y, leftlane_x, 3)

    if right_lane_window_coord_list:
        rightlane_x = [x for x, _ in right_lane_window_coord_list]
        rightlane_y = [y for _, y in right_lane_window_coord_list]

        if len(rightlane_x) >= window_number and len(rightlane_y) >= window_number:
            right_fit = polyfit(rightlane_y, rightlane_x, 3)

    return right_fit, left_fit


def evaluate_polynomial(coefficients: List[float], x: float) -> float:
    result = 0.0
    power = len(coefficients) - 1
    for coeff in coefficients:
        result += coeff * x**power
        power -= 1
    return result

def calculate_centre_lane_px(left_fit: np.ndarray, right_fit: np.ndarray, y_eval: int) -> float:
    bottom_x_right = evaluate_polynomial(right_fit, y_eval)
    bottom_x_left = evaluate_polynomial(left_fit, y_eval)
    return bottom_x_left + (bottom_x_right - bottom_x_left) / 2

def calculate_centre_lane_mid_height_px_right(right_lane_window_coord_list: List[Tuple[int, int]], window_height: int, distance: int, left_lane: bool) -> Tuple[float, float]:
    x_coord_right2, y_coord_right2 = right_lane_window_coord_list[window_height]
    x_coord_right, y_coord_right = right_lane_window_coord_list[window_height - 2]
    lane_slope = (y_coord_right2 - x_coord_right) / (x_coord_right2 - x_coord_right)
    if lane_slope == 0:
        lane_slope = 0.0001
    perpendicular_slope = -1 / lane_slope

    if left_lane:
        perpendicular_line_x = x_coord_right + (distance / np.sqrt(1 + perpendicular_slope ** 2))
        perpendicular_line_y = y_coord_right + (
                perpendicular_slope * (distance / np.sqrt(1 + perpendicular_slope ** 2)))
    else:
        perpendicular_line_x = x_coord_right - (distance / np.sqrt(1 + perpendicular_slope ** 2))
        perpendicular_line_y = y_coord_right - (
                perpendicular_slope * (distance / np.sqrt(1 + perpendicular_slope ** 2)))

    return perpendicular_line_x, perpendicular_line_y

def calculate_centre_lane_mid_height_px_left(left_lane_window_coord_list: List[Tuple[int, int]], window_height: int, distance: int, right_lane: bool) -> Tuple[float, float]:
    x_coord_left, y_coord_left = left_lane_window_coord_list[window_height]
    x_coord_left2, y_coord_left2 = left_lane_window_coord_list[window_height - 2]
    lane_slope = (y_coord_left2 - y_coord_left) / (x_coord_left2 - x_coord_left)
    if lane_slope == 0:
        lane_slope = 0.0001
    perpendicular_slope = -1 / lane_slope

    if right_lane:
        perpendicular_line_x = x_coord_left - (distance / np.sqrt(1 + perpendicular_slope ** 2))
        perpendicular_line_y = y_coord_left - (
                perpendicular_slope * (distance / np.sqrt(1 + perpendicular_slope ** 2)))
    else:
        perpendicular_line_x = x_coord_left + (distance / np.sqrt(1 + perpendicular_slope ** 2))
        perpendicular_line_y = y_coord_left + (
                perpendicular_slope * (distance / np.sqrt(1 + perpendicular_slope ** 2)))

    return perpendicular_line_x, perpendicular_line_y

def calculate_car_position(image_middle: float, centre_lane_px: float) -> float:
    centre_offset_pixels = image_middle - centre_lane_px
    return centre_offset_pixels + centre_lane_px

def calculate_theta_in_deg(centre_lane_mid_height_px: float, centre_lane_mid_height_py: float, car_position: float, y_eval: int) -> float:
    delta_x = np.abs(centre_lane_mid_height_px - car_position)
    delta_y = y_eval - centre_lane_mid_height_py

    if centre_lane_mid_height_px < car_position:
        theta_rad = math.atan2(delta_y, delta_x) - np.pi / 2
        return theta_rad * 180 / np.pi
    elif centre_lane_mid_height_px > car_position:
        theta_rad = math.atan2(-delta_y, delta_x) + np.pi / 2
        return theta_rad * 180 / np.pi
    else:
        return 0

def calculate_steering_angle(left_lane_window_coord_list: List[Tuple[int, int]],
                             right_lane_window_coord_list: List[Tuple[int, int]],
                             right_fit: np.ndarray,
                             left_fit: np.ndarray,
                             warped: np.ndarray) -> float:
    window_nr: int = 12
    window_height: int = 11
    distance_to_lane: int = 290
    y_eval: int = 600
    image_middle: float = warped.shape[1] / 2
    centre_lane_mid_height_px: float = None
    centre_lane_mid_height_py: float = None
    left_lane: bool = False
    right_lane: bool = False

    if left_lane_window_coord_list is None and right_lane_window_coord_list is None:
        return 0

    if len(left_lane_window_coord_list) <= window_nr and len(right_lane_window_coord_list) <= window_nr:
        return 0

    if len(left_lane_window_coord_list) >= window_nr and len(right_lane_window_coord_list) >= window_nr:
        reference_list.append("m")
        x_coord_right, y_coord_right = right_lane_window_coord_list[window_height]
        x_coord_left, y_coord_left = left_lane_window_coord_list[window_height]
        centre_lane_mid_height_py = y_coord_right
        centre_lane_mid_height_px = x_coord_left + (x_coord_right - x_coord_left) / 2
        centre_lane_px: float = calculate_centre_lane_px(left_fit, right_fit, y_eval)

    elif len(right_lane_window_coord_list) > window_nr:
        left_lane = False
        distance: int = 300

        x_coord_start, y_coord_start = right_lane_window_coord_list[0]
        if x_coord_start <= ranges[2]:
            reference_list.append("r2")
        elif x_coord_start > ranges[2]:
            reference_list.append("r3")

        if len(reference_list) > 2:
            if reference_list[-3] == "l0" and reference_list[-2] == "l1" and reference_list[-1] == "r2":
                left_lane = True
                reference_list[-1] = "l2"
            elif reference_list[-2] == "l2" and reference_list[-1] == "r2":
                left_lane = True
                reference_list[-1] = "l2"
            elif reference_list[-3] == "l2" and reference_list[-2] == "l2" and reference_list[-1] == "r2":
                left_lane = False

        centre_lane_mid_height_px, centre_lane_mid_height_py = calculate_centre_lane_mid_height_px_right(right_lane_window_coord_list, window_height, distance, left_lane)
        bottom_x_right = evaluate_polynomial(right_fit, y_eval)
        centre_lane_px: float = bottom_x_right - distance_to_lane

    elif len(left_lane_window_coord_list) > window_nr:
        right_lane = False
        distance: int = 300

        x_coord_start, y_coord_start = left_lane_window_coord_list[0]
        if x_coord_start <= ranges[0]:
            reference_list.append("l0")
        elif x_coord_start > ranges[0]:
            reference_list.append("l1")

        if len(reference_list) > 2:
            if reference_list[-3] == "r3" and reference_list[-2] == "r2" and reference_list[-1] == "l1":
                right_lane = True
                reference_list[-1] = "r1"
            elif reference_list[-2] == "r1" and reference_list[-1] == "l1":
                right_lane = True
                reference_list[-1] = "r1"
            elif reference_list[-3] == "r1" and reference_list[-2] == "r1" and reference_list[-1] == "l1":
                right_lane = False

        centre_lane_mid_height_px, centre_lane_mid_height_py = calculate_centre_lane_mid_height_px_left(left_lane_window_coord_list, window_height, distance, right_lane)
        bottom_x_left = evaluate_polynomial(left_fit, y_eval)
        centre_lane_px: float = bottom_x_left - distance_to_lane

    if centre_lane_px is None:
        return 0

    car_position: float = calculate_car_position(image_middle, centre_lane_px)
    return calculate_theta_in_deg(centre_lane_mid_height_px, centre_lane_mid_height_py, car_position, y_eval)


def pipeline() -> None:
    image_directory: str = "determine_dir_opt/calced"
    image_files: List[str] = os.listdir(image_directory)
    total_processing_time: float = 0.0

    png_files: List[str] = [file for file in image_files if file.endswith('.png')]
    image_paths_doubled: List[str] = png_files

    random.shuffle(image_paths_doubled)
    num_images: int = len(image_paths_doubled)

    for image_file in image_paths_doubled:
        image_path: str = os.path.join(image_directory, image_file)
        original_frame: np.ndarray = cv2.imread(image_path)
        print(image_path)

        start_time: float = time.time()
        gray_image: np.ndarray = colortogreyImage(original_frame)
        warped: np.ndarray = birdseyeView(gray_image)
        cannyLines: np.ndarray = detectlaneLines(warped)
        dilated_edges: np.ndarray = dilatelaneEdges(cannyLines)
        left_lane_window_coord_list, right_lane_window_coord_list = detect_lane_pixels(dilated_edges, warped)
        right_fit, left_fit = polynomalFitThree(left_lane_window_coord_list, right_lane_window_coord_list)
        steering_angle: float = calculate_steering_angle(left_lane_window_coord_list, right_lane_window_coord_list, right_fit, left_fit, warped)
        end_time: float = time.time()
        processing_time: float = end_time - start_time
        total_processing_time += processing_time

        if False:
            # Extract the angle from the filename
            extracted_angle: float = float(image_file[:-4])  # Assuming the angle is at the beginning and the extension is ".png"

            # Compare the extracted angle with the calculated angle with tolerance
            if abs(extracted_angle - steering_angle) <= 1.0:
                print(f"Angles match! Given: {extracted_angle} vs calced: {steering_angle}")
            else:
                print(f"Angles do not match. Given: {extracted_angle} vs calced: {steering_angle}")

    avg_processing_time: float = total_processing_time / num_images
    images_per_second: float = 1. / avg_processing_time
    print(f"Processed {num_images} images in an average of {avg_processing_time:.5f} seconds.")
    print(f"Images per second: {images_per_second:.2f}")


if __name__ == '__main__':
    pipeline()
    cv2.destroyAllWindows()
