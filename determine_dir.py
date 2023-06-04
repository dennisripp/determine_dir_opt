import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
from enum import Enum
import time

enable_debug = False
enable_histogram = False
sliding_window_enbaled = False
enable_print = False
enable_pipeline = True

ranges = [160, 320, 480, 640]
reference_list = []

# Define conversions in x and y from pixels space to meters
ym_per_pix: float = 1 / 2250  # meters per pixel in y dimension
xm_per_pix: float = 1 / 2250  # meters per pixel in x dimension


def colortogreyImage(original_frame):
    image_gray: numpy.ndarray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    if enable_debug:
        cv2.imshow('image_gray', image_gray)
        cv2.waitKey(0)
    return image_gray


def birdseyeView(image_gray):
    width: int = image_gray.shape[1]
    #print("width", width)
    height: int = image_gray.shape[0]
    # print("height", height)
    # Rois

    # Polygon I chose per example images:
    source: numpy.ndarray = np.float32([[(20, 440), (100, 180), (590, 180), (640, 470)]])
    # Straightened Polygon:
    destination: numpy.ndarray = np.float32([[0, 480], [0, 0], [640, 0], [640, 480]])
    # Transform to birds eye view
    # transform matrix given source and destination points
    matrix: numpy.ndarray = cv2.getPerspectiveTransform(source, destination)
    # applies perspective transformation to the image
    warped: numpy.ndarray = cv2.warpPerspective(image_gray, matrix, (width, height), flags=cv2.INTER_LINEAR)
    if enable_debug:
        cv2.imshow('Birds Eye', warped)
        cv2.waitKey(0)

    return warped

def detectlaneLines(warped):
    threshold_low: int = 100
    threshold_high: int = 200
    # python problem, sometimes variables are seen as global
    image_copy: numpy.ndarray = np.copy(warped)
    # blur
    image_blurred: numpy.ndarray = cv2.GaussianBlur(image_copy, (7, 7), 0)
    # Canny
    image_canny: numpy.ndarray = cv2.Canny(image_blurred, threshold_low, threshold_high)
    if enable_debug:
        cv2.imshow('binary_image', image_canny)
        cv2.waitKey(0)
    return image_canny


def dilatelaneEdges(image_canny):
    image = np.copy(image_canny)
    # edges = cv2.Canny(image, threshold1=180, threshold2=200)

    # define kernel for dilation operation
    kernel: numpy.ndarray = np.ones((5, 5), np.uint8)

    # apply dilation on the edges
    dilated_edges: numpy.ndarray = cv2.dilate(image, kernel, iterations=1)

    if enable_debug:
        cv2.imshow('Dilated Edges', dilated_edges)
        cv2.waitKey(0)
    return dilated_edges


def detect_lane_pixels(dilated_edges: numpy.ndarray, warped: numpy.ndarray):
    binary_image = np.copy(dilated_edges)
    nwindows: int = 20  # number of sliding windows
    window_length: int = 70  # sliding window length
    window_height: numpy.int8 = np.int8(binary_image.shape[0] // nwindows)  # sliding window height
    minpix: int = 20  # threshold of pixels for a positiv lane find

    # bottom_half
    # Get distribution of pixel intensities along the horizontal axis of the bottom half of the image
    # binary_image.shape[0] * 3// 4: selects the 3/4 rows of the image along the y-axis
    # shape[0]: gives the number of rows (height)
    # \: : all columns are included
    bottom_fourth: numpy.ndarray = binary_image[binary_image.shape[0] * 7 // 8:, :]
    # histogram
    # np.sum: adds all pixel values in each column of lower picture
    histogram: numpy.ndarray = np.sum(bottom_fourth, axis=0)  # histogram

    # plot the histogram
    if enable_histogram:
        # Create a line plot of the pixel intensities distribution
        plt.plot(histogram)
        # Add labels and a title to the plot
        plt.xlabel('Column')
        plt.ylabel('Pixel Intensity')
        plt.title('Distribution of Pixel Intensities along the Horizontal Axis')
        # Show the plot
        plt.show()

    # Convert the image back to rgb in order to draw colored boxes
    # image to draw boxes on
    imagewithBoxes = warped.copy()
    # find indices of all non-zero elements in the binary thresholded image
    # binary image: pixels in lanes are set to white and black
    # with other words, pixels that belong to the lane lines are stored in row and column indices
    nonzero: tuple = binary_image.nonzero()
    nonzeroy: numpy.ndarray = np.array(nonzero[0])
    nonzerox: numpy.ndarray = np.array(nonzero[1])
    # calculate the index of the middle row
    # Split histogram into left and right halves
    midpoint: int = histogram.shape[0] // 2
    # Define the range for the left and right halves
    left_range_start: int = 0
    left_range_end: int = midpoint - 0  # Adjust the end point as desired

    right_range_start: int = midpoint + 0  # Adjust the start point as desired
    right_range_end: int = histogram.shape[0]

    # Split histogram into left and right halves
    left_histogram: numpy.ndarray = histogram[left_range_start:left_range_end]
    right_histogram: numpy.ndarray = histogram[right_range_start:right_range_end]

    # Find the peak pixel intensities in each half of the histogram
    threshold_histogramm = 500
    leftx_peak_mask: numpy.ndarray = left_histogram > threshold_histogramm
    rightx_peak_mask: numpy.ndarray = right_histogram > threshold_histogramm
    leftx_current = None
    rightx_current = None
    if np.any(leftx_peak_mask):
        # Left lane, high intensities indicate white pixels as starting point of lane
        # x coordinate for lane detection in the bottom half of the image
        leftx_current: numpy.int64 = np.argmax(left_histogram * leftx_peak_mask)
        if enable_debug:
            print('x coord. for left lane', leftx_current)
    if np.any(rightx_peak_mask):
        # right lane
        rightx_current: numpy.int64 = np.argmax(right_histogram * rightx_peak_mask) + midpoint
        if enable_debug:
            print('x coord. for right lane', rightx_current)

    left_lane: list = []
    right_lane: list = []
    right_lane_window_coord_list = []
    left_lane_window_coord_list = []
    # Identify window boundaries for x and y in left and right lane
    left_pixels_found = True
    right_pixels_found = True
    for window in range(nwindows):

        win_y_low: numpy.int32 = dilated_edges.shape[0] - (window + 1) * window_height
        win_y_high: numpy.int32 = dilated_edges.shape[0] - window * window_height
        # left lane
        if leftx_current is not None and left_pixels_found is True:
            win_xleft_low: numpy.int64 = leftx_current - window_length
            win_xleft_high: numpy.int64 = leftx_current + window_length

            # Identify the nonzero pixels in x and y within the window
            left_x: numpy.ndarray = nonzerox[
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)]
            left_y: numpy.ndarray = nonzeroy[
                (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                        nonzerox < win_xleft_high)]
            if len(left_x) > 0 and len(left_y) > 0:  # this line is new
                # If you found > minpix pixels, recenter next window on their mean position
                if len(left_x) > minpix and len(left_y) > minpix:
                    leftx_current = np.int32(np.mean(left_x))
                    lefty_current = np.int32(np.mean(left_y))
                    left_window_coord_xy = (leftx_current, lefty_current)
                    left_lane_window_coord_list.append(left_window_coord_xy)
                    # Draw the windows on the visualization image
                    if sliding_window_enbaled:
                        cv2.rectangle(imagewithBoxes, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                                      (0, 255, 0), 4)
            else:
                left_pixels_found = False

        # right lane
        if rightx_current is not None and right_pixels_found is True:
            win_xright_low = rightx_current - window_length
            win_xright_high = rightx_current + window_length
            # Draw the windows on the visualization image

            right_x = nonzerox[(nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)]
            right_y = nonzeroy[(nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)]

            if len(right_x) > 0 and len(right_y) > 0:
                if len(right_x) > minpix and len(right_y) > minpix:
                    rightx_current = np.int32(np.mean(right_x))
                    righty_current = np.int32(np.mean(right_y))
                    right_window_coord_xy = (rightx_current, righty_current)
                    right_lane_window_coord_list.append(right_window_coord_xy)
                    if sliding_window_enbaled:
                        cv2.rectangle(imagewithBoxes, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                                      (0, 255, 0), 4)
            else:
                right_pixels_found = False
    # shows the grey image with sliding windows
    if sliding_window_enbaled:
        cv2.imshow('Boxes', imagewithBoxes)
        cv2.waitKey(0)

    return left_lane_window_coord_list, right_lane_window_coord_list


class Curvature(Enum):
    CURVED = 1
    STRAIGHT = 2
    UNDETERMINED = 3


class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    UNDETERMINED = 3



def polynomalFitThree(left_lane_window_coord_list, right_lane_window_coord_list):
    # coefficients decreasing order of degree
    # y = a * x^2 + b * x + c
    left_fit = None
    right_fit = None

    window_number = 12  # number 1 is the window at the bottom of the image

    if left_lane_window_coord_list is not None and len(left_lane_window_coord_list) > 1:
        leftlane_x = [x for x, _ in left_lane_window_coord_list]
        leftlane_y = [y for _, y in left_lane_window_coord_list]

        if len(leftlane_x) >= window_number and len(leftlane_y) >= window_number:
            left_fit = np.polyfit(leftlane_y, leftlane_x, 3)
            if enable_print:
                print('coeffs 3rd-order left lane', left_fit)

    if right_lane_window_coord_list:
        rightlane_x = [x for x, _ in right_lane_window_coord_list]
        rightlane_y = [y for _, y in right_lane_window_coord_list]

        if len(rightlane_x) >= window_number and len(rightlane_y) >= window_number:
            right_fit = np.polyfit(rightlane_y, rightlane_x, 3)
            if enable_print:
                print('coeffs 3rd-order right lane', right_fit)

    return right_fit, left_fit


def calculate_steering_angle(left_lane_window_coord_list, right_lane_window_coord_list, right_fit, left_fit, warped):
    # important: if anything goes wrong here steering angle will be zero
    # needs averaging over more images
    # versatz nach links oder rechts
    window_nr = 12
    window_height = 11
    distance_to_lane = 290

    theta_in_deg = None
    steering_angle: int = 0
    # height of the image
    image_height: int = warped.shape[0]
    # what ii says
    image_middle: int = warped.shape[1] / 2
    # y coord:
    y_eval = 600  # y-coordinate of the bottom of the image, at the moment lowered by 200 px
    delta_y = None
    # coordinates derived from windows concerning lane positions in image
    x_coord_left = None
    x_coord_right = None
    y_coord_right = None
    y_coord_left = None
    # x coordinates of the lanes at the bottom of the image, ergo where the lanes start
    bottom_x_right = None
    bottom_x_left = None
    centre_lane_px = None
    centre_lane_mid_height_px = None
    centre_lane_mid_height_py = None
    lane_slope = None
    perpendicular_slope = None
    perpendicular_line_x = None
    perpendicular_line_y = None
    perpendicular_line_x2 = None
    perpendicular_line_y2 = None
    slope_positive_x = None
    slope_positive_y = None
    slope_negative_x = None
    slope_negative_y = None

    #############
    # if there are no window coordinates for steering angle
    if left_lane_window_coord_list is None and right_lane_window_coord_list is None:
        return 0
    # if both lanes are short return
    elif len(left_lane_window_coord_list) <= window_nr and len(right_lane_window_coord_list) <= window_nr:
        return 0
    # determine current and previous positions:
    # if we have both lanes
    elif len(left_lane_window_coord_list) >= window_nr and len(right_lane_window_coord_list) >= window_nr:
        if len(reference_list) > 3:
            reference_list.pop(0)
        reference_list.append("m")

        x_coord_right, y_coord_right = right_lane_window_coord_list[window_height]
        x_coord_left, y_coord_left = left_lane_window_coord_list[window_height]
        centre_lane_mid_height_py = y_coord_right
        centre_lane_mid_height_px = x_coord_left + (x_coord_right - x_coord_left) / 2
        bottom_x_right = right_fit[0] * y_eval ** 3 + right_fit[1] * y_eval ** 2 + right_fit[2] * y_eval + right_fit[3]
        bottom_x_left = left_fit[0] * y_eval ** 3 + left_fit[1] * y_eval ** 2 + left_fit[2] * y_eval + left_fit[3]
        centre_lane_px = bottom_x_left + (bottom_x_right - bottom_x_left) / 2

    elif len(right_lane_window_coord_list) > window_nr:
        # check for lane shift
        left_lane = False
        # Distance from the lane line
        distance = 300

        x_coord_start, y_coord_start = right_lane_window_coord_list[0]
        if len(reference_list) > 3:
            reference_list.pop(0)
        if x_coord_start <= ranges[2]:
            reference_list.append("r2")
        elif x_coord_start > ranges[2]:
            reference_list.append("r3")
        if len(reference_list) > 2:
            if reference_list[-3] == "l0" and reference_list[-2] == "l1" and reference_list[-1] == "r2":  # right shift
                left_lane = True
                reference_list[-1] = "l2"
            elif reference_list[-2] == "l2" and reference_list[-1] == "r2":
                left_lane = True
                reference_list[-1] = "l2"
            elif reference_list[-3] == "l2" and reference_list[-2] == "l2" and reference_list[-1] == "r2":
                left_lane = False

        x_coord_right2, y_coord_right2 = right_lane_window_coord_list[
            window_height]  # Get the x and y coordinates from the tuple
        x_coord_right, y_coord_right = right_lane_window_coord_list[window_height - 2]

        ###right lane is right lane###
        lane_slope = (y_coord_right2 - x_coord_right) / (x_coord_right2 - x_coord_right)
        if lane_slope == 0:
            lane_slope = 0.0001
        perpendicular_slope = -1 / lane_slope
        # Calculate the perpendicular line coordinates at the desired distance
        perpendicular_line_x = x_coord_right - (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
        perpendicular_line_y = y_coord_right - (perpendicular_slope * (distance / (
            np.sqrt(1 + perpendicular_slope ** 2))))
        centre_lane_mid_height_px = perpendicular_line_x
        centre_lane_mid_height_py = perpendicular_line_y

        ###right lane is left lane###
        if left_lane is True:
            # Calculate the perpendicular line coordinates at the desired distance
            perpendicular_line_x = x_coord_right + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            perpendicular_line_y = y_coord_right + perpendicular_slope * distance / (
                np.sqrt(1 + perpendicular_slope ** 2))
            new_x = x_coord_right + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            new_y = y_coord_right + (perpendicular_slope * (distance / (
                np.sqrt(1 + perpendicular_slope ** 2))))
            centre_lane_mid_height_px = perpendicular_line_x
            centre_lane_mid_height_py = perpendicular_line_y

        ####x Coord at the bottom##########
        bottom_x_right = right_fit[0] * y_eval ** 3 + right_fit[1] * y_eval ** 2 + right_fit[2] * y_eval + right_fit[3]
        centre_lane_px = bottom_x_right - distance_to_lane


    elif len(left_lane_window_coord_list) > window_nr:
        right_lane = False
        # Distance from the lane line
        distance = 300
        x_coord_start, y_coord_start = left_lane_window_coord_list[0]
        if len(reference_list) > 3:
            reference_list.pop(0)
        if x_coord_start <= ranges[0]:
            reference_list.append("l0")
        elif x_coord_start > ranges[0]:
            reference_list.append("l1")
        if len(reference_list) > 2:
            if reference_list[-3] == "r3" and reference_list[-2] == "r2" and reference_list[-1] == "l1":  # right shift
                right_lane = True
                reference_list[-1] = "r1"
            elif reference_list[-2] == "r1" and reference_list[-1] == "l1":
                right_lane = True
                reference_list[-1] = "r1"
            elif reference_list[-3] == "r1" and reference_list[-2] == "r1" and reference_list[-1] == "l1":
                right_lane = False

        x_coord_left, y_coord_left = left_lane_window_coord_list[
            window_height]  # Get the x and y coordinates from the tuple
        x_coord_left2, y_coord_left2 = left_lane_window_coord_list[window_height - 2]
        # Calculate the slope of the lane line
        lane_slope = (y_coord_left2 - y_coord_left) / (x_coord_left2 - x_coord_left)
        if lane_slope == 0:
            lane_slope = 0.0001
        perpendicular_slope = -1 / lane_slope

        # Calculate the perpendicular line coordinates at the desired distance
        perpendicular_line_x = x_coord_left + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
        perpendicular_line_y = y_coord_left + perpendicular_slope * distance / (np.sqrt(1 + perpendicular_slope ** 2))
        centre_lane_mid_height_px = perpendicular_line_x
        centre_lane_mid_height_py = perpendicular_line_y

        ##left lane is a right lane##
        if right_lane is True:
            # Calculate the perpendicular line coordinates at the desired distance
            perpendicular_line_x = x_coord_left - (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            perpendicular_line_y = y_coord_left - (perpendicular_slope * (distance / (
                np.sqrt(1 + perpendicular_slope ** 2))))
            centre_lane_mid_height_px = perpendicular_line_x
            centre_lane_mid_height_py = perpendicular_line_y

        ####x Coord at the bottom##########
        bottom_x_left = left_fit[0] * y_eval ** 3 + left_fit[1] * y_eval ** 2 + left_fit[2] * y_eval + left_fit[3]
        centre_lane_px = bottom_x_left - distance_to_lane

    if centre_lane_px is None:
        return 0
    # continue calculation
    # calculate actual car position
    centre_offset_pixels = image_middle - centre_lane_px
    car_position = centre_offset_pixels + centre_lane_px

    # dx und dy: Distances between the coord. in pixel
    # dx:
    delta_x: np.float64 = abs(centre_lane_mid_height_px - car_position)
    delta_y = y_eval - centre_lane_mid_height_py

    # Winkelberechnung zwischen zwei Punkten;
    # First determine direction, then calculate, easier for angle towards vertical line
    if centre_lane_mid_height_px < car_position:
        theta_rad: float = math.atan2(delta_y, delta_x) - np.pi / 2
        theta_in_deg: float = theta_rad * 180 / np.pi
        # print('The steering angle to the left: ', theta_in_deg)

    elif centre_lane_mid_height_px > car_position:
        theta_rad: float = math.atan2(-delta_y, delta_x) + np.pi / 2
        theta_in_deg: float = theta_rad * 180 / np.pi
        # print('The steering angle to the right: ', theta_in_deg)

    elif centre_lane_mid_height_px == car_position:
        theta_in_deg = 0

    steering_angle = theta_in_deg

    return steering_angle





def calculate_steering_angle(left_lane_window_coord_list, right_lane_window_coord_list, right_fit, left_fit, warped):
    # important: if anything goes wrong here steering angle will be zero
    # needs averaging over more images
    # versatz nach links oder rechts
    window_nr = 12
    window_height = 11
    distance_to_lane = 290

    theta_in_deg = None
    steering_angle: int = 0
    # height of the image
    image_height: int = warped.shape[0]
    # what ii says
    image_middle: int = warped.shape[1] / 2
    # y coord:
    y_eval = 600  # y-coordinate of the bottom of the image, at the moment lowered by 200 px
    delta_y = None
    # coordinates derived from windows concerning lane positions in image
    x_coord_left = None
    x_coord_right = None
    y_coord_right = None
    y_coord_left = None
    # x coordinates of the lanes at the bottom of the image, ergo where the lanes start
    bottom_x_right = None
    bottom_x_left = None
    centre_lane_px = None
    centre_lane_mid_height_px = None
    centre_lane_mid_height_py = None
    lane_slope = None
    perpendicular_slope = None
    perpendicular_line_x = None
    perpendicular_line_y = None
    perpendicular_line_x2 = None
    perpendicular_line_y2 = None
    slope_positive_x = None
    slope_positive_y = None
    slope_negative_x = None
    slope_negative_y = None

    #############
    # if there are no window coordinates for steering angle
    if left_lane_window_coord_list is None and right_lane_window_coord_list is None:
        return 0
    # if both lanes are short return
    elif len(left_lane_window_coord_list) <= window_nr and len(right_lane_window_coord_list) <= window_nr:
        return 0
    # determine current and previous positions:
    # if we have both lanes
    elif len(left_lane_window_coord_list) >= window_nr and len(right_lane_window_coord_list) >= window_nr:
        if len(reference_list) > 3:
            reference_list.pop(0)
        reference_list.append("m")

        x_coord_right, y_coord_right = right_lane_window_coord_list[window_height]
        x_coord_left, y_coord_left = left_lane_window_coord_list[window_height]
        centre_lane_mid_height_py = y_coord_right
        centre_lane_mid_height_px = x_coord_left + (x_coord_right - x_coord_left) / 2
        bottom_x_right = right_fit[0] * y_eval ** 3 + right_fit[1] * y_eval ** 2 + right_fit[2] * y_eval + right_fit[3]
        bottom_x_left = left_fit[0] * y_eval ** 3 + left_fit[1] * y_eval ** 2 + left_fit[2] * y_eval + left_fit[3]
        centre_lane_px = bottom_x_left + (bottom_x_right - bottom_x_left) / 2

    elif len(right_lane_window_coord_list) > window_nr:
        # check for lane shift
        left_lane = False
        # Distance from the lane line
        distance = 300

        x_coord_start, y_coord_start = right_lane_window_coord_list[0]
        if len(reference_list) > 3:
            reference_list.pop(0)
        if x_coord_start <= ranges[2]:
            reference_list.append("r2")
        elif x_coord_start > ranges[2]:
            reference_list.append("r3")
        if len(reference_list) > 2:
            if reference_list[-3] == "l0" and reference_list[-2] == "l1" and reference_list[-1] == "r2":  # right shift
                left_lane = True
                reference_list[-1] = "l2"
            elif reference_list[-2] == "l2" and reference_list[-1] == "r2":
                left_lane = True
                reference_list[-1] = "l2"
            elif reference_list[-3] == "l2" and reference_list[-2] == "l2" and reference_list[-1] == "r2":
                left_lane = False

        x_coord_right2, y_coord_right2 = right_lane_window_coord_list[
            window_height]  # Get the x and y coordinates from the tuple
        x_coord_right, y_coord_right = right_lane_window_coord_list[window_height - 2]

        ###right lane is right lane###
        lane_slope = (y_coord_right2 - x_coord_right) / (x_coord_right2 - x_coord_right)
        if lane_slope == 0:
            lane_slope = 0.0001
        perpendicular_slope = -1 / lane_slope
        # Calculate the perpendicular line coordinates at the desired distance
        perpendicular_line_x = x_coord_right - (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
        perpendicular_line_y = y_coord_right - (perpendicular_slope * (distance / (
            np.sqrt(1 + perpendicular_slope ** 2))))
        centre_lane_mid_height_px = perpendicular_line_x
        centre_lane_mid_height_py = perpendicular_line_y
        if enable_debug:
            perpendicular_line_x2 = x_coord_right + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            perpendicular_line_y2 = y_coord_right + (perpendicular_slope * (distance / (
                np.sqrt(1 + perpendicular_slope ** 2))))
            # Steigungsgerade
            slope_positive_x = x_coord_right + (distance / np.sqrt(1 + lane_slope ** 2))
            slope_positive_y = y_coord_right + (lane_slope * (distance / np.sqrt(1 + lane_slope ** 2)))

            slope_negative_x = x_coord_right - (distance / np.sqrt(1 + lane_slope ** 2))
            slope_negative_y = y_coord_right - (lane_slope * (distance / np.sqrt(1 + lane_slope ** 2)))

        ###right lane is left lane###
        if left_lane is True:
            # Calculate the perpendicular line coordinates at the desired distance
            perpendicular_line_x = x_coord_right + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            perpendicular_line_y = y_coord_right + perpendicular_slope * distance / (
                np.sqrt(1 + perpendicular_slope ** 2))
            new_x = x_coord_right + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            new_y = y_coord_right + (perpendicular_slope * (distance / (
                np.sqrt(1 + perpendicular_slope ** 2))))
            centre_lane_mid_height_px = perpendicular_line_x
            centre_lane_mid_height_py = perpendicular_line_y
            if enable_debug:
                perpendicular_line_x2 = x_coord_right - (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
                perpendicular_line_y2 = y_coord_right - (perpendicular_slope * (distance / (
                    np.sqrt(1 + perpendicular_slope ** 2))))
                # Steigungsgerade
                slope_positive_x = x_coord_right + (distance / np.sqrt(1 + lane_slope ** 2))
                slope_positive_y = y_coord_right + (lane_slope * (distance / np.sqrt(1 + lane_slope ** 2)))
                slope_negative_x = x_coord_right - (distance / np.sqrt(1 + lane_slope ** 2))
                slope_negative_y = y_coord_right - (lane_slope * (distance / np.sqrt(1 + lane_slope ** 2)))

        ####x Coord at the bottom##########
        bottom_x_right = right_fit[0] * y_eval ** 3 + right_fit[1] * y_eval ** 2 + right_fit[2] * y_eval + right_fit[3]
        centre_lane_px = bottom_x_right - distance_to_lane

        if enable_debug:
            # Draw Steigung und Orthogonale
            cv2.line(warped, (int(perpendicular_line_x2), int(perpendicular_line_y2)),
                     (int(perpendicular_line_x), int(perpendicular_line_y)),
                     (255, 0, 0), 2)
            # Steigung
            cv2.line(warped, (int(slope_negative_x), int(slope_negative_y)),
                     (int(slope_positive_x), int(slope_positive_y)),
                     (255, 0, 0), 5)

    elif len(left_lane_window_coord_list) > window_nr:
        right_lane = False
        # Distance from the lane line
        distance = 300
        x_coord_start, y_coord_start = left_lane_window_coord_list[0]
        if len(reference_list) > 3:
            reference_list.pop(0)
        if x_coord_start <= ranges[0]:
            reference_list.append("l0")
        elif x_coord_start > ranges[0]:
            reference_list.append("l1")
        if len(reference_list) > 2:
            if reference_list[-3] == "r3" and reference_list[-2] == "r2" and reference_list[-1] == "l1":  # right shift
                right_lane = True
                reference_list[-1] = "r1"
            elif reference_list[-2] == "r1" and reference_list[-1] == "l1":
                right_lane = True
                reference_list[-1] = "r1"
            elif reference_list[-3] == "r1" and reference_list[-2] == "r1" and reference_list[-1] == "l1":
                right_lane = False

        x_coord_left, y_coord_left = left_lane_window_coord_list[
            window_height]  # Get the x and y coordinates from the tuple
        x_coord_left2, y_coord_left2 = left_lane_window_coord_list[window_height - 2]
        # Calculate the slope of the lane line
        lane_slope = (y_coord_left2 - y_coord_left) / (x_coord_left2 - x_coord_left)
        if lane_slope == 0:
            lane_slope = 0.0001
        perpendicular_slope = -1 / lane_slope

        # Calculate the perpendicular line coordinates at the desired distance
        perpendicular_line_x = x_coord_left + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
        perpendicular_line_y = y_coord_left + perpendicular_slope * distance / (np.sqrt(1 + perpendicular_slope ** 2))
        centre_lane_mid_height_px = perpendicular_line_x
        centre_lane_mid_height_py = perpendicular_line_y
        if enable_debug:
            perpendicular_line_x2 = x_coord_left - (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            perpendicular_line_y2 = y_coord_left - perpendicular_slope * distance / (
                np.sqrt(1 + perpendicular_slope ** 2))
            # Steigungsgerade
            slope_positive_x = x_coord_left + (distance / np.sqrt(1 + lane_slope ** 2))
            slope_positive_y = y_coord_left + (lane_slope * distance / np.sqrt(1 + lane_slope ** 2))
            slope_negative_x = x_coord_left - (distance / np.sqrt(1 + lane_slope ** 2))
            slope_negative_y = y_coord_left - (lane_slope * distance / np.sqrt(1 + lane_slope ** 2))
        ##left lane is a right lane##
        if right_lane is True:
            # Calculate the perpendicular line coordinates at the desired distance
            perpendicular_line_x = x_coord_left - (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
            perpendicular_line_y = y_coord_left - (perpendicular_slope * (distance / (
                np.sqrt(1 + perpendicular_slope ** 2))))
            centre_lane_mid_height_px = perpendicular_line_x
            centre_lane_mid_height_py = perpendicular_line_y
            if enable_debug:
                perpendicular_line_x2 = x_coord_left + (distance / (np.sqrt(1 + perpendicular_slope ** 2)))
                perpendicular_line_y2 = y_coord_left + (perpendicular_slope * (distance / (
                    np.sqrt(1 + perpendicular_slope ** 2))))
                # Steigungsgerade
                slope_positive_x = x_coord_left + (distance / np.sqrt(1 + lane_slope ** 2))
                slope_positive_y = y_coord_left + (lane_slope * (distance / np.sqrt(1 + lane_slope ** 2)))

                slope_negative_x = x_coord_left - (distance / np.sqrt(1 + lane_slope ** 2))
                slope_negative_y = y_coord_left - (lane_slope * (distance / np.sqrt(1 + lane_slope ** 2)))

        ####x Coord at the bottom##########
        bottom_x_left = left_fit[0] * y_eval ** 3 + left_fit[1] * y_eval ** 2 + left_fit[2] * y_eval + left_fit[3]
        centre_lane_px = bottom_x_left - distance_to_lane

        if enable_debug:
            # Draw the lane line
            cv2.line(warped, (int(perpendicular_line_x2), int(perpendicular_line_y2)),
                     (int(perpendicular_line_x), int(perpendicular_line_y)),
                     (255, 0, 0), 2)
            cv2.line(warped, (int(slope_negative_x), int(slope_negative_y)),
                     (int(slope_positive_x), int(slope_positive_y)),
                     (255, 0, 0), 5)

    if centre_lane_px is None:
        return 0
    # continue calculation
    # calculate actual car position
    centre_offset_pixels = image_middle - centre_lane_px
    car_position = centre_offset_pixels + centre_lane_px

    # dx und dy: Distances between the coord. in pixel
    # dx:
    delta_x: numpy.float64 = abs(centre_lane_mid_height_px - car_position)
    delta_y = y_eval - centre_lane_mid_height_py

    # Winkelberechnung zwischen zwei Punkten;
    # First determine direction, then calculate, easier for angle towards vertical line
    if centre_lane_mid_height_px < car_position:
        theta_rad: float = math.atan2(delta_y, delta_x) - np.pi / 2
        theta_in_deg: float = theta_rad * 180 / np.pi
        #print('The steering angle to the left: ', theta_in_deg)

    elif centre_lane_mid_height_px > car_position:
        theta_rad: float = math.atan2(-delta_y, delta_x) + np.pi / 2
        theta_in_deg: float = theta_rad * 180 / np.pi
        #print('The steering angle to the right: ', theta_in_deg)

    elif centre_lane_mid_height_px == car_position:
        theta_in_deg = 0

    steering_angle = theta_in_deg

    if enable_debug:
        # print angle on image
        car_position = int(car_position)
        start_point = (car_position, y_eval)

        centre_lane_mid_height_px = int(centre_lane_mid_height_px)
        centre_lane_mid_height_py = int(centre_lane_mid_height_py)
        end_point = (centre_lane_mid_height_px, centre_lane_mid_height_py)
        color = (255, 0, 0)
        thickness = 3
        image_with_line = cv2.line(warped, start_point, end_point, color, thickness)
        cv2.imshow("image with line", image_with_line)
        cv2.waitKey(0)

    if enable_print:
        print("x_coord_left, y_coord_left: ", x_coord_left, y_coord_left)
        print('centre_lane_mid_height_px', centre_lane_mid_height_px)
        print('centre_lane_px: ', centre_lane_px)
        print('centre_offset_pixels: ', centre_offset_pixels)
        print('car_position: ', car_position)
        print('reference_list: ', reference_list)
        print('steering_angle: ', steering_angle)

    return steering_angle


def pipeline():
    image_directory = "calced/"
    image_files = os.listdir(image_directory)
    total_processing_time = 0.0
    # Filter the list of files to include only .png files
    png_files = [file for file in image_files if file.endswith('.png')]

    import random

    # Double the image paths
    image_paths_doubled = png_files

    # Shuffle the image paths
    random.shuffle(image_paths_doubled)
    num_images = len(image_paths_doubled)


    for image_file in image_paths_doubled:
        image_path = os.path.join(image_directory, image_file)
        original_frame = cv2.imread(image_path)

        # Preprocessing and main processing
        start_time = time.time()
        gray_image = colortogreyImage(original_frame)
        warped = birdseyeView(gray_image)
        cannyLines = detectlaneLines(warped)
        dilated_edges = dilatelaneEdges(cannyLines)
        left_lane_window_coord_list, right_lane_window_coord_list = detect_lane_pixels(dilated_edges, warped)
        right_fit, left_fit = polynomalFitThree(left_lane_window_coord_list, right_lane_window_coord_list)
        steering_angle = calculate_steering_angle(left_lane_window_coord_list, right_lane_window_coord_list, right_fit, left_fit, warped)
        end_time = time.time()
        processing_time = end_time - start_time
        total_processing_time += processing_time

        # Extract the angle from the filename
        extracted_angle = float(image_file[:-4])  # Assuming the angle is at the beginning and the extension is ".png"

        # Compare the extracted angle with the calculated angle with tolerance
        if abs(extracted_angle - steering_angle) <= 1.0:
            print(f"Angles match! Given: {extracted_angle} vs calced: {steering_angle}")
        else:
            print("Angles do not match.")


    avg_processing_time = total_processing_time / num_images
    images_per_second = 1. / avg_processing_time
    print(f"Processed {num_images} images in an average of {avg_processing_time:.5f} seconds.")
    print(f"Images per second: {images_per_second:.2f}")


if __name__ == '__main__':
    pipeline()

    cv2.destroyAllWindows()