/* Sample Main*/

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <string>
#include <random>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <cmath>
#include <cassert>
#include <opencv2/core.hpp>


float ym_per_pix = 1 / 2250.0f;
float xm_per_pix = 1 / 2250.0f;
std::vector<int> ranges = {160, 320, 480, 640};
std::vector<std::string> reference_list;

cv::Mat colortogreyImage(cv::Mat original_frame) {
    cv::Mat gray_frame;
    cv::cvtColor(original_frame, gray_frame, cv::COLOR_BGR2GRAY);
    return gray_frame;
}

cv::Mat birdseyeView(cv::Mat image_gray) {
    int width = image_gray.cols;
    int height = image_gray.rows;
    cv::Point2f source[4] = {cv::Point2f(20, 440), cv::Point2f(100, 180), cv::Point2f(590, 180), cv::Point2f(640, 470)};
    cv::Point2f destination[4] = {cv::Point2f(0, 480), cv::Point2f(0, 0), cv::Point2f(640, 0), cv::Point2f(640, 480)};

    cv::Mat matrix = cv::getPerspectiveTransform(source, destination);
    cv::Mat warped;
    cv::warpPerspective(image_gray, warped, matrix, cv::Size(width, height), cv::INTER_LINEAR);
    return warped;
}

cv::Mat detectlaneLines(cv::Mat warped) {
    int threshold_low = 100;
    int threshold_high = 200;
    cv::Mat image_blurred;
    cv::GaussianBlur(warped, image_blurred, cv::Size(7, 7), 0);
    cv::Mat edges;
    cv::Canny(image_blurred, edges, threshold_low, threshold_high);
    return edges;
}

cv::Mat dilatelaneEdges(cv::Mat image_canny) {
    cv::Mat dilated_edges;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(image_canny, dilated_edges, kernel, cv::Point(-1, -1), 1);
    return dilated_edges;
}

Eigen::VectorXd solve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    return A.householderQr().solve(b);
}

std::vector<float> polyfit(const std::vector<float>& src_y, const std::vector<float>& src_x, int order) {
    assert(src_x.size() == src_y.size());
    assert(src_x.size() >= order + 1);

    std::vector<double> t_double(src_x.begin(), src_x.end());
    std::vector<double> v_double(src_y.begin(), src_y.end());

    Eigen::MatrixXd T(src_x.size(), order + 1);
    Eigen::VectorXd V = Eigen::VectorXd::Map(&v_double.front(), v_double.size());
    Eigen::VectorXd result;

    for (size_t i = 0; i < src_x.size(); ++i) {
        for (size_t j = 0; j < order + 1; ++j) {
            T(i, j) = std::pow(t_double.at(i), j);
        }
    }

    result = solve(T, V);

    std::vector<float> coeff(order + 1);
    for (int k = 0; k < order + 1; k++) {
        coeff[k] = static_cast<float>(result[k]);
    }

    // Reverse elements of coeff
    int num_elements = coeff.size();
    for (int i = 0; i < num_elements / 2; ++i) {
        float temp = coeff[i];
        coeff[i] = coeff[num_elements - 1 - i];
        coeff[num_elements - 1 - i] = temp;
    }

    return coeff;
}



std::pair<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>> detect_lane_pixels(
    cv::Mat dilated_edges, cv::Mat warped) {
    cv::Mat binary_image = dilated_edges.clone();
    int nwindows = 20;
    int window_length = 70;
    int window_height = binary_image.rows / nwindows;

    cv::Mat bottom_fourth = binary_image.rowRange(binary_image.rows * 7 / 8, binary_image.rows);
    cv::Mat sum_result;
    cv::reduce(bottom_fourth, sum_result, 0, cv::REDUCE_SUM, CV_32F);
    cv::Mat histogram = cv::Mat::zeros(1, sum_result.cols, CV_32FC1);
    sum_result.convertTo(histogram, CV_32FC1);

    cv::Mat nonzero;
    cv::findNonZero(binary_image, nonzero);
    cv::Mat nonzeroy, nonzerox;
    nonzeroy.create(nonzero.rows, 1, CV_32SC1);
    nonzerox.create(nonzero.rows, 1, CV_32SC1);
    for (int i = 0; i < nonzero.total(); i++) {
        nonzeroy.at<int>(i, 0) = nonzero.at<cv::Point>(i).y;
        nonzerox.at<int>(i, 0) = nonzero.at<cv::Point>(i).x;
    }

    int midpoint = histogram.cols / 2;

    int left_range_start = 0;
    int left_range_end = midpoint - 0;
    int right_range_start = midpoint + 0;
    int right_range_end = histogram.cols;

    cv::Mat left_histogram = histogram.colRange(left_range_start, left_range_end);
    cv::Mat right_histogram = histogram.colRange(right_range_start, right_range_end);

    int threshold_histogramm = 500;
    cv::Mat leftx_peak_mask = left_histogram > threshold_histogramm;
    cv::Mat rightx_peak_mask = right_histogram > threshold_histogramm;

    double minVal, maxVal;
    cv::Point minLoc, maxLoc, leftx_current, rightx_current;
    cv::minMaxLoc(left_histogram, &minVal, &maxVal, &minLoc, &maxLoc, leftx_peak_mask);
    leftx_current = cv::countNonZero(leftx_peak_mask) > 0 ? maxLoc : cv::Point(-1, -1);

    cv::minMaxLoc(right_histogram, &minVal, &maxVal, &minLoc, &maxLoc, rightx_peak_mask);
    rightx_current = cv::countNonZero(rightx_peak_mask) > 0 ? cv::Point(maxLoc.x + midpoint, maxLoc.y) : cv::Point(-1, -1);

    std::vector<std::pair<int, int>> left_lane_window_coord_list;
    std::vector<std::pair<int, int>> right_lane_window_coord_list;

    for (int window = 0; window < nwindows; window++) {
        int win_y_low = binary_image.rows - (window + 1) * window_height;
        int win_y_high = binary_image.rows - window * window_height;

        if (leftx_current.x != -1) {
            int win_xleft_low = leftx_current.x - window_length;
            int win_xleft_high = leftx_current.x + window_length;

            cv::Mat left_mask = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high);

            if (cv::countNonZero(left_mask) > 0) {
                cv::Scalar mean = cv::mean(nonzerox, left_mask);
                int leftx_current_new = cvRound(mean[0]);
                int lefty_current_new = cvRound(cv::mean(nonzeroy, left_mask)[0]);
                left_lane_window_coord_list.push_back(std::make_pair(leftx_current_new, lefty_current_new));
                leftx_current.x = leftx_current_new;
            }
        }

        if (rightx_current.x != -1) {
            int win_xright_low = rightx_current.x - window_length;
            int win_xright_high = rightx_current.x + window_length;

            cv::Mat right_mask = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high);

            if (cv::countNonZero(right_mask) > 0) {
                cv::Scalar mean = cv::mean(nonzerox, right_mask);
                int rightx_current_new = cvRound(mean[0]);
                int righty_current_new = cvRound(cv::mean(nonzeroy, right_mask)[0]);
                right_lane_window_coord_list.push_back(std::make_pair(rightx_current_new, righty_current_new));
                rightx_current.x = rightx_current_new;
            }
        }
    }

    return std::make_pair(left_lane_window_coord_list, right_lane_window_coord_list);
}



std::pair<std::vector<float>, std::vector<float>> polynomialFitThree(
    const std::vector<std::pair<int, int>>& left_lane_window_coord_list,
    const std::vector<std::pair<int, int>>& right_lane_window_coord_list) {
    std::vector<float> left_fit, right_fit;
    int window_number = 12;

    if (!left_lane_window_coord_list.empty() && left_lane_window_coord_list.size() > 1) {
        std::vector<float> leftlane_x, leftlane_y;
        for (const auto& coord : left_lane_window_coord_list) {
            leftlane_x.push_back(coord.first);
            leftlane_y.push_back(coord.second);
        }

        if (leftlane_x.size() >= window_number && leftlane_y.size() >= window_number) {
            left_fit = polyfit(leftlane_y, leftlane_x, 3);
            cv::Mat left_fit_mat(left_fit, CV_32F);
            left_fit = std::vector<float>(left_fit_mat.begin<float>(), left_fit_mat.end<float>());
        }
    }

    if (!right_lane_window_coord_list.empty()) {
        std::vector<float> rightlane_x, rightlane_y;
        for (const auto& coord : right_lane_window_coord_list) {
            rightlane_x.push_back(coord.first);
            rightlane_y.push_back(coord.second);
        }

        if (rightlane_x.size() >= window_number && rightlane_y.size() >= window_number) {
            right_fit = polyfit(rightlane_y, rightlane_x, 3);
            cv::Mat right_fit_mat(right_fit, CV_32F);
            right_fit = std::vector<float>(right_fit_mat.begin<float>(), right_fit_mat.end<float>());
        }
    }

    return std::make_pair(right_fit, left_fit);
}

float evaluate_polynomial(const std::vector<float>& coefficients, float x) {
    float result = 0.0;
    int power = coefficients.size() - 1;
    for (const float& coeff : coefficients) {
        result += coeff * pow(x, power);
        power -= 1;
    }
    return result;
}

float calculate_centre_lane_px(std::vector<float> left_fit, std::vector<float> right_fit, int y_eval) {
    float bottom_x_right = evaluate_polynomial(right_fit, y_eval);
    float bottom_x_left = evaluate_polynomial(left_fit, y_eval);
    return bottom_x_left + (bottom_x_right - bottom_x_left) / 2;
}

std::pair<float, float> calculate_centre_lane_mid_height_px_right(
    const std::vector<std::pair<int, int>>& right_lane_window_coord_list, int window_height, int distance, bool left_lane) {
    int x_coord_right2 = right_lane_window_coord_list[window_height].first;
    int y_coord_right2 = right_lane_window_coord_list[window_height].second;
    int x_coord_right = right_lane_window_coord_list[window_height - 2].first;
    int y_coord_right = right_lane_window_coord_list[window_height - 2].second;

    float lane_slope = static_cast<float>(y_coord_right2 - x_coord_right) / static_cast<float>(x_coord_right2 - x_coord_right);
    if (lane_slope == 0) {
        lane_slope = 0.0001;
    }
    float perpendicular_slope = -1 / lane_slope;

    float perpendicular_line_x, perpendicular_line_y;
    if (left_lane) {
        perpendicular_line_x = x_coord_right + (distance / sqrt(1 + pow(perpendicular_slope, 2)));
        perpendicular_line_y = y_coord_right + (perpendicular_slope * (distance / sqrt(1 + pow(perpendicular_slope, 2))));
    } else {
        perpendicular_line_x = x_coord_right - (distance / sqrt(1 + pow(perpendicular_slope, 2)));
        perpendicular_line_y = y_coord_right - (perpendicular_slope * (distance / sqrt(1 + pow(perpendicular_slope, 2))));
    }

    return std::make_pair(perpendicular_line_x, perpendicular_line_y);
}

std::pair<float, float> calculate_centre_lane_mid_height_px_left(
    const std::vector<std::pair<int, int>>& left_lane_window_coord_list, int window_height, int distance, bool right_lane) {
    int x_coord_left = left_lane_window_coord_list[window_height].first;
    int y_coord_left = left_lane_window_coord_list[window_height].second;
    int x_coord_left2 = left_lane_window_coord_list[window_height - 2].first;
    int y_coord_left2 = left_lane_window_coord_list[window_height - 2].second;

    float lane_slope = static_cast<float>(y_coord_left2 - y_coord_left) / static_cast<float>(x_coord_left2 - x_coord_left);
    if (lane_slope == 0) {
        lane_slope = 0.0001;
    }
    float perpendicular_slope = -1 / lane_slope;

    float perpendicular_line_x, perpendicular_line_y;
    if (right_lane) {
        perpendicular_line_x = x_coord_left - (distance / sqrt(1 + pow(perpendicular_slope, 2)));
        perpendicular_line_y = y_coord_left - (perpendicular_slope * (distance / sqrt(1 + pow(perpendicular_slope, 2))));
    } else {
        perpendicular_line_x = x_coord_left + (distance / sqrt(1 + pow(perpendicular_slope, 2)));
        perpendicular_line_y = y_coord_left + (perpendicular_slope * (distance / sqrt(1 + pow(perpendicular_slope, 2))));
    }

    return std::make_pair(perpendicular_line_x, perpendicular_line_y);
}

float calculate_car_position(float image_middle, float centre_lane_px) {
    float centre_offset_pixels = image_middle - centre_lane_px;
    return centre_offset_pixels + centre_lane_px;
}

float calculate_theta_in_deg(float centre_lane_mid_height_px, float centre_lane_mid_height_py, float car_position, int y_eval) {
    float delta_x = fabs(centre_lane_mid_height_px - car_position);
    float delta_y = y_eval - centre_lane_mid_height_py;

    if (centre_lane_mid_height_px < car_position) {
        float theta_rad = atan2(delta_y, delta_x) - M_PI / 2;
        return theta_rad * 180 / M_PI;
    } else if (centre_lane_mid_height_px > car_position) {
        float theta_rad = atan2(-delta_y, delta_x) + M_PI / 2;
        return theta_rad * 180 / M_PI;
    } else {
        return 0;
    }
}

float calculate_steering_angle(
    const std::vector<std::pair<int, int>>& left_lane_window_coord_list,
    const std::vector<std::pair<int, int>>& right_lane_window_coord_list,
    const std::vector<float>& right_fit,
    const std::vector<float>& left_fit,
    const cv::Mat& warped) {
    int window_nr = 12;
    int window_height = 11;
    int distance_to_lane = 290;
    int y_eval = 600;
    float image_middle = static_cast<float>(warped.cols) / 2;
    float centre_lane_mid_height_px = 0.0f;
    float centre_lane_mid_height_py = 0.0f;
    bool left_lane = false;
    bool right_lane = false;
    float centre_lane_px = 0.0f; // Initialize centre_lane_px

    if (left_lane_window_coord_list.empty() && right_lane_window_coord_list.empty()) {
        return 0;
    }

    if (left_lane_window_coord_list.size() <= window_nr && right_lane_window_coord_list.size() <= window_nr) {
        return 0;
    }

    if (left_lane_window_coord_list.size() >= window_nr && right_lane_window_coord_list.size() >= window_nr) {
        reference_list.push_back("m");
        int x_coord_right = right_lane_window_coord_list[window_height].first;
        int y_coord_right = right_lane_window_coord_list[window_height].second;
        int x_coord_left = left_lane_window_coord_list[window_height].first;
        int y_coord_left = left_lane_window_coord_list[window_height].second;
        centre_lane_mid_height_py = y_coord_right;
        centre_lane_mid_height_px = x_coord_left + (x_coord_right - x_coord_left) / 2;
        centre_lane_px = calculate_centre_lane_px(left_fit, right_fit, y_eval);
    } else if (right_lane_window_coord_list.size() > window_nr) {
        left_lane = false;
        int distance = 300;

        int x_coord_start = right_lane_window_coord_list[0].first;
        int y_coord_start = right_lane_window_coord_list[0].second;
        if (x_coord_start <= ranges[2]) {
            reference_list.push_back("r2");
        } else if (x_coord_start > ranges[2]) {
            reference_list.push_back("r3");
        }

        if (reference_list.size() > 2) {
            if (reference_list[reference_list.size() - 3] == "l0" && reference_list[reference_list.size() - 2] == "l1" && reference_list[reference_list.size() - 1] == "r2") {
                left_lane = true;
                reference_list[reference_list.size() - 1] = "l2";
            } else if (reference_list[reference_list.size() - 2] == "l2" && reference_list[reference_list.size() - 1] == "r2") {
                left_lane = true;
                reference_list[reference_list.size() - 1] = "l2";
            } else if (reference_list[reference_list.size() - 3] == "l2" && reference_list[reference_list.size() - 2] == "l2" && reference_list[reference_list.size() - 1] == "r2") {
                left_lane = false;
            }
        }

        std::pair<float, float> result = calculate_centre_lane_mid_height_px_right(right_lane_window_coord_list, window_height, distance, left_lane);
        centre_lane_mid_height_px = result.first;
        centre_lane_mid_height_py = result.second;
        float bottom_x_right = evaluate_polynomial(right_fit, y_eval);
        centre_lane_px = bottom_x_right - distance_to_lane;
    } else if (left_lane_window_coord_list.size() > window_nr) {
        right_lane = false;
        int distance = 300;

        int x_coord_start = left_lane_window_coord_list[0].first;
        int y_coord_start = left_lane_window_coord_list[0].second;
        if (x_coord_start <= ranges[0]) {
            reference_list.push_back("l0");
        } else if (x_coord_start > ranges[0]) {
            reference_list.push_back("l1");
        }

        if (reference_list.size() > 2) {
            if (reference_list[reference_list.size() - 3] == "r3" && reference_list[reference_list.size() - 2] == "r2" && reference_list[reference_list.size() - 1] == "l1") {
                right_lane = true;
                reference_list[reference_list.size() - 1] = "r1";
            } else if (reference_list[reference_list.size() - 2] == "r1" && reference_list[reference_list.size() - 1] == "l1") {
                right_lane = true;
                reference_list[reference_list.size() - 1] = "r1";
            } else if (reference_list[reference_list.size() - 3] == "r1" && reference_list[reference_list.size() - 2] == "r1" && reference_list[reference_list.size() - 1] == "l1") {
                right_lane = false;
            }
        }

        std::pair<float, float> result = calculate_centre_lane_mid_height_px_left(left_lane_window_coord_list, window_height, distance, right_lane);
        centre_lane_mid_height_px = result.first;
        centre_lane_mid_height_py = result.second;
        float bottom_x_left = evaluate_polynomial(left_fit, y_eval);
        centre_lane_px = bottom_x_left - distance_to_lane;
    }

    if (centre_lane_px == 0.0f) {
        return 0;
    }

    float car_position = calculate_car_position(image_middle, centre_lane_px);
    return calculate_theta_in_deg(centre_lane_mid_height_px, centre_lane_mid_height_py, car_position, y_eval);
}


void pipeline() {

    std::list<std::string> paths = {
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/-14.3113.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/0.0000.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/2.5240.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/-2.9560.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/44.6374.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/13.9685.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/11.9468.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/46.6398.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/20.8183.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/21.4592.png",
        "/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/calced/-3.4595.png"
    };
    int desiredCount = 3600;
    int numImages = paths.size();
    float totalProcessingTime = 0.0f;

    // Multiply the paths to reach the desired count
    while (numImages < desiredCount) {
        for (const std::string& path : paths) {
            paths.push_back(path);
            numImages++;

            if (numImages >= desiredCount) {
                break;
            }
        }
    }

    // Displaying the paths
    for (const std::string& path : paths) {
        std::size_t lastSlashIndex = path.find_last_of('/');
        std::string filename = path.substr(lastSlashIndex + 1);
        cv::Mat original_frame = cv::imread(path);
        auto start_time = std::chrono::system_clock::now();
        cv::Mat gray_image = colortogreyImage(original_frame);
        cv::Mat warped = birdseyeView(gray_image);
        cv::Mat cannyLines = detectlaneLines(warped);
        cv::Mat dilated_edges = dilatelaneEdges(cannyLines);

        auto [left_lane_window_coord_list, right_lane_window_coord_list] = detect_lane_pixels(dilated_edges, warped);

        auto [right_fit, left_fit] = polynomialFitThree(left_lane_window_coord_list, right_lane_window_coord_list);

        float steering_angle = calculate_steering_angle(left_lane_window_coord_list, right_lane_window_coord_list, right_fit, left_fit, warped);

        // std::cout << steering_angle<< std::endl;
        // std::cout << filename << std::endl;
        // std::cout << "#############################" << std::endl;

        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<float> duration = end_time - start_time;
        float processingTime = duration.count();
        totalProcessingTime += processingTime;
    }
       // Calculate and display the average processing time
    float averageProcessingTime = totalProcessingTime / numImages;
    float imagesPerSecond = 1.0f / averageProcessingTime;
    std::cout << "Average processing time: " << averageProcessingTime << " seconds" << std::endl;
    std::cout << "Images processed per second: " << imagesPerSecond << std::endl;

    // cv::Mat original_frame = cv::imread("/home/dennis/Desktop/VSCode/petra_git/determine_direction_sliding_window/translate/frame.png");
    // int i = 0;
    // auto start_time = std::chrono::system_clock::now();
    // cv::Mat gray_image = colortogreyImage(original_frame);
    // cv::Mat warped = birdseyeView(gray_image);
    // cv::Mat cannyLines = detectlaneLines(warped);
    // cv::Mat dilated_edges = dilatelaneEdges(cannyLines);
    // std::cout << ++i<< std::endl;

    // auto [left_lane_window_coord_list, right_lane_window_coord_list] = detect_lane_pixels(dilated_edges, warped);
    // std::cout << ++i << std::endl;

    // auto [right_fit, left_fit] = polynomialFitThree(left_lane_window_coord_list, right_lane_window_coord_list);
    // std::cout << ++i<< std::endl;

    // float steering_angle = calculate_steering_angle(left_lane_window_coord_list, right_lane_window_coord_list, right_fit, left_fit, warped);
    // std::cout << ++i<< std::endl;

    // std::cout << steering_angle<< std::endl;
}



int main() {
    pipeline();

    // // Beispielbild laden
    // cv::Mat inputImage = cv::imread("frame.jpg");

    // // Debug-Daten Struktur erstellen
    // DebugData* debugData  = new DebugData();
    // cv::Mat imageAfterGRB2Gray;
    // cv::Mat imageCanny;
    // cv::Mat imageEdges;
    // cv::Mat slidingWindow;
    // Direction direction_lane;
    // Direction direction_curvature;
    // Curvature curv;
    // std::pair<cv::Mat, cv::Mat> poly_coeff;
    // std::pair<cv::Mat, cv::Mat> poly_coeff_rw;

    // debugData->imageBRG2Gray = &imageAfterGRB2Gray;
    // debugData->imageCanny = &imageCanny;
    // debugData->imageEdges = &imageEdges;
    // debugData->polynomalFit = &poly_coeff;
    // debugData->slidingWindow = &slidingWindow;
    // debugData->laneDirection = &direction_lane;
    // debugData->curveDirection = &direction_curvature;
    // debugData->laneCurvature = &curv;
    // debugData->polynomalFit_rw = &poly_coeff_rw;

    // // Winkel berechnen und Debug-Bilder speichern
    // int angle = processor.calculateAngle(inputImage, debugData);

    // std::cout << "##############################################" << std::endl;

    // // Ergebnisse ausgeben
    // std::cout << "coeff right fit: " << poly_coeff.first << std::endl;
    // std::cout << "coeff right rw fit: " << poly_coeff_rw.first << std::endl;
    // std::cout << "\n\n" << std::endl;
    // std::cout << "coeff left fit: " << poly_coeff.second << std::endl;
    // std::cout << "coeff left rw fit: " << poly_coeff_rw.second << std::endl;
    // std::cout << "\n\n" << std::endl;
    // std::cout << "Winkel: " << angle << std::endl;    
    // std::cout << "slope : " << getDirectionString(direction_lane) << std::endl;
    // std::cout << "curv dir: " << getDirectionString(direction_curvature) << std::endl;
    // std::cout << "curv: " << getCurvatureString(curv) << std::endl;
    // cv::imshow("image_after_preprocess.jpg", imageAfterGRB2Gray);
    // cv::imshow("image_with_sliding_window.jpg", imageCanny);
    // cv::imshow("image_with_fitted_lines.jpg", imageEdges);
    // cv::imshow("slidingwindow.jpg", slidingWindow);
    // cv::waitKey(0);


    return 0;
}