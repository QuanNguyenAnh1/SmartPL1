#include "ImageProcessor.h"
ImageProcessor::ImageProcessor() {
}

ImageProcessor::~ImageProcessor() {
}

cv::Mat ImageProcessor::preprocessImage( cv::Mat& image) {
    int xMargin = 2;  
    int yMargin = 2;  
    int height = image.rows;
    int width = image.cols;
    cv::Rect roiRect(xMargin, yMargin, width - 2 * xMargin, height - 2 * yMargin);
    cv::Mat roi = image(roiRect);

    cv::Mat grayImage;
    cv::cvtColor(roi, grayImage, cv::COLOR_BGR2GRAY);

    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);

    cv::Mat edges;
    cv::Canny(blurredImage, edges, 50, 150);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat closing;
    cv::morphologyEx(edges, closing, cv::MORPH_CLOSE, kernel);

    cv::Mat dilated;
    cv::dilate(closing, dilated, kernel, cv::Point(-1, -1), 1);

    return dilated;
}

std::vector<cv::Point2f> ImageProcessor::sortPointsClockwise( std::vector<cv::Point2f>& points) {
    // Sort points clockwise implementation
    std::vector<cv::Point2f> sortedPoints = points;

    // Sort the points based on their x-coordinates
    std::sort(sortedPoints.begin(), sortedPoints.end(), []( cv::Point2f& a,  cv::Point2f& b) {
        return a.x < b.x;
    });

    // Separate points into top and bottom halves
    std::vector<cv::Point2f> top(sortedPoints.begin(), sortedPoints.begin() + 2);
    std::vector<cv::Point2f> bottom(sortedPoints.begin() + 2, sortedPoints.end());

    // Sort top points by y-coordinate to find the top-left point first
    std::sort(top.begin(), top.end(), []( cv::Point2f& a,  cv::Point2f& b) {
        return a.y < b.y;
    });

    // Sort bottom points by y-coordinate to find the bottom-left point first
    std::sort(bottom.begin(), bottom.end(), []( cv::Point2f& a,  cv::Point2f& b) {
        return a.y > b.y;
    });

    // Combine top-left, top-right, bottom-right, bottom-left
    std::vector<cv::Point2f> result;
    result.push_back(top[0]);
    result.push_back(top[1]);
    result.push_back(bottom[0]);
    result.push_back(bottom[1]);

    return result;
}

std::vector<cv::Point2f> ImageProcessor::selectLargestQuadrilateral( std::vector<cv::Point2f>& points) {
    // Select largest quadrilateral implementation
    float maxArea = 0;
    std::vector<cv::Point2f> quad;

    size_t numPoints = points.size();

    // Try all combinations of four points to find the one with the largest area
    for (size_t i = 0; i < numPoints; ++i) {
        for (size_t j = i + 1; j < numPoints; ++j) {
            for (size_t k = j + 1; k < numPoints; ++k) {
                for (size_t l = k + 1; l < numPoints; ++l) {
                    std::vector<cv::Point2f> polygon = { points[i], points[j], points[k], points[l] };
                    cv::Mat hull(polygon);
                    double area = cv::contourArea(hull);  // Calculate the area
                    if (area > maxArea) {
                        maxArea = area;
                        quad = polygon;
                    }
                }
            }
        }
    }

    return quad;
}

std::vector<cv::Point2f> ImageProcessor::findQuadrilateralFromPoints( std::vector<cv::Point2f>& points) {
    // Find quadrilateral from points implementation
    std::vector<cv::Point2f> hullPoints;

    // Ensure that points form a convex polygon
    cv::Mat hull(points);
    cv::convexHull(hull, hullPoints);

    // Now we need to ensure that we have exactly four points (for a quadrilateral)
    if (hullPoints.size() < 4) {
        // Not enough points for a quadrilateral
        return std::vector<cv::Point2f>(); // Returning empty vector
    }
    else if (hullPoints.size() > 4) {
        // Too many points, we need to select the best four that form the largest area
        hullPoints = selectLargestQuadrilateral(hullPoints);
    }

    // Sort the hull points to order them clockwise
    hullPoints = sortPointsClockwise(hullPoints);

    return hullPoints;
}

std::vector<cv::Vec4i> ImageProcessor::detectLines( cv::Mat& edged) {
    // Detect lines using HoughLinesP
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edged, lines, 1, CV_PI / 180, 100, 100, 50);
    return lines;
}

cv::Mat ImageProcessor::visualizeLines( cv::Mat& image,  std::vector<cv::Vec4i>& lines) {
    cv::Mat lineImage = image.clone();

    if (!lines.empty()) {
        for ( cv::Vec4i& line : lines) {
            cv::Point pt1(line[0], line[1]);
            cv::Point pt2(line[2], line[3]);
            cv::line(lineImage, pt1, pt2, cv::Scalar(0, 255, 0), 2);  // Draw lines in green
        }
    }

    return lineImage;
}
std::vector<cv::Point2f> ImageProcessor::clusterAndValidateIntersections( std::vector<cv::Point2f>& intersections,  cv::Mat& image) {
    cv::Mat intersection_points(intersections.size(), 2, CV_32F);
    for (size_t i = 0; i < intersections.size(); ++i) {
        intersection_points.at<float>(i, 0) = intersections[i].x;
        intersection_points.at<float>(i, 1) = intersections[i].y;
    }

    // Clustering with K-Means
    int k = 500; // Considering only one cluster
    cv::Mat labels, centers;
    kmeans(intersection_points, k, labels, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

    // Get centroids of clusters
    std::vector<cv::Point2f> centroids;
    for (int i = 0; i < centers.rows; ++i) {
        centroids.emplace_back(cv::Point2f(centers.at<float>(i, 0), centers.at<float>(i, 1)));
    }

    // Validation
    std::vector<cv::Point2f> validated_corners;
    for ( auto& pt : centroids) {
        if (pt.x > 0 && pt.y > 0 && pt.x < image.cols && pt.y < image.rows) {
            validated_corners.push_back(pt);
        }
    }

    return validated_corners;
}
std::vector<cv::Point2f> ImageProcessor::lineIntersections( std::vector<cv::Vec4i>& lines,  cv::Mat& image) {
    // Line intersections implementation
    std::vector<cv::Point2f> intersections;

    for (size_t i = 0; i < lines.size(); ++i) {
        for (size_t j = i + 1; j < lines.size(); ++j) {
            float x1 = lines[i][0], y1 = lines[i][1];
            float x2 = lines[i][2], y2 = lines[i][3];
            float x3 = lines[j][0], y3 = lines[j][1];
            float x4 = lines[j][2], y4 = lines[j][3];

            float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
            if (denom != 0) {
                float px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
                float py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
                intersections.push_back(cv::Point2f(px, py));
            }
        }
    }

    // Apply clustering and validation
    std::vector<cv::Point2f> filteredIntersections = clusterAndValidateIntersections(intersections, image);
    return filteredIntersections;
}


cv::Mat ImageProcessor::visualizeIntersections( cv::Mat& image,  std::vector<cv::Point2f>& intersections) {
    
    cv::Mat intersectionImage = image.clone();

    for ( cv::Point2f& pt : intersections) {
        cv::circle(intersectionImage, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)), 5, cv::Scalar(0, 0, 255), -1);  // Draw intersections in red
    }

    return intersectionImage;
}

void ImageProcessor::findQuadrilateralFromPointsAndSave( std::vector<cv::Point2f>& points,  std::string& imageName) {
    // Find quadrilateral from points implementation
    std::ofstream outputFile(imageName + ".txt");
    if (outputFile.is_open()) {
        for ( auto& point : points) {
            outputFile << point.x << " " << point.y << std::endl;
        }
        outputFile.close();
    }
}

std::vector<cv::Point2f> ImageProcessor::processImageAndDetectQuadrilateral( cv::Mat& image) {
    // Process image and detect quadrilateral implementation
    cv::Mat edged = preprocessImage(image);
    std::vector<cv::Vec4i> lines = detectLines(edged);
    cv::Mat imageLined = visualizeLines(image, lines);
    // cv::imshow("Result0", imageLined);
    // cv::waitKey(0);
    std::vector<cv::Point2f> intersections = lineIntersections(lines, image);
    cv::Mat intersectionImage = visualizeIntersections(image, intersections);
    // cv::imshow("Result1", intersectionImage);
    // cv::waitKey(0);
    std::vector<cv::Point2f> srcPts;// = clusterAndValidateIntersections(intersections, image);
    srcPts = findQuadrilateralFromPoints(intersections);
    
    // if (!srcPts.empty()) {
    //     findQuadrilateralFromPointsAndSave(srcPts, "result_" + std::to_string(time(nullptr)));
    // }
    return srcPts;
}

cv::Mat ImageProcessor::drawPoints(cv::Mat image, std::vector<cv::Point2f> points){
    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),  // Màu đỏ
        cv::Scalar(0, 255, 0),  // Màu xanh lá
        cv::Scalar(255, 0, 0),  // Màu xanh dương
        cv::Scalar(0, 255, 255) // Màu vàng
    };

    for (size_t i = 0; i < points.size(); ++i) {
        cv::Point2f point = points[i];
        cv::Scalar color = colors[i % colors.size()];

        cv::circle(image, point, 10, color, -1);
    }
    return image;
}

cv::Mat ImageProcessor::mergeImages( std::vector<cv::Mat>& images) {
    cv::Mat complete_img = cv::Mat::zeros(images[0].size(), images[0].type());
    int part_h = images[0].rows / 2;
    int part_w = images[0].cols / 3;

    auto get_image_index = [&](int i, int j,  std::vector<cv::Mat>& images,  std::vector<std::pair<int, int>>& part_centers) {
        std::vector<double> distances;
        for ( auto& center : part_centers) {
            double distance = std::sqrt(std::pow(i - center.first, 2) + std::pow(j - center.second, 2));
            distances.push_back(distance);
        }
        int min_distance_index = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
        if (images[min_distance_index].at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {
            distances[min_distance_index] = 1000;
            min_distance_index = std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));
        }
        return min_distance_index;
    };

    std::vector<std::pair<int, int>> part_centers;
    for (int row = 0; row < 2; row++) {
        for (int col = 0; col < 3; col++) {
            int center_i = row * part_h + part_h / 2;
            int center_j = col * part_w + part_w / 2;
            part_centers.emplace_back(center_i, center_j);
        }
    }

    for (int i = 0; i < images[0].rows; i++) {
        for (int j = 0; j < images[0].cols; j++) {
            int index = get_image_index(i, j, images, part_centers);
            complete_img.at<cv::Vec3b>(i, j) = images[index].at<cv::Vec3b>(i, j);
        }
    }

    return complete_img;
}

std::vector<cv::Mat> ImageProcessor::readImagesFromFolder( std::string& folderPath){
    std::vector<cv::Mat> images;
    for (int i = 1; i <= 6; ++i) {
        std::string imageFileName = folderPath + "/" + std::to_string(i) + ".jpg";
        cv::Mat inputImg = cv::imread(imageFileName);
        if( inputImg.empty() )
        {
            std::cout << "Could not open or find the image: " << imageFileName << std::endl;
            continue;
        }
        images.push_back(inputImg);
    }
    return images;
}

cv::Mat ImageProcessor::blur_image(cv::Mat image) {
    cv::Mat blurred_image = image.clone();
    std::vector<int> kernel_sizes = {3, 3};

    for (size_t i = 0; i < kernel_sizes.size(); ++i) {
        cv::GaussianBlur(blurred_image, blurred_image, cv::Size(kernel_sizes[i], kernel_sizes[i]), 0);
    }

    return blurred_image;
}

cv::Mat ImageProcessor::edge_detect(cv::Mat image) {
    cv::Mat blurred_image;
    cv::GaussianBlur(image, blurred_image, cv::Size(3, 3), 0);

    cv::Mat gray_image;
    cv::cvtColor(blurred_image, gray_image, cv::COLOR_BGR2GRAY);

    cv::Mat thresh;
    cv::threshold(gray_image, thresh, 170, 255, cv::THRESH_BINARY);

    cv::Mat blurred_thresh = ImageProcessor::blur_image(thresh);

    cv::Mat edges;
    cv::Canny(blurred_thresh, edges, 100, 250, 3);

    cv::Mat closing_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat closing;
    cv::morphologyEx(edges, closing, cv::MORPH_CLOSE, closing_kernel);

    cv::Mat kernel = (cv::Mat_<uchar>(3, 3) << 0, 1, 0,
                                               1, 1, 1,
                                               0, 1, 0);

    cv::Mat dilated;
    cv::dilate(closing, dilated, kernel, cv::Point(-1, -1), 1);

    return dilated;
}

cv::Mat ImageProcessor::visualize_lines(cv::Mat image, std::vector<cv::Vec4i>& lines) {
    if (!lines.empty()) {
        for (const auto& line : lines) {
            int x1 = line[0];
            int y1 = line[1];
            int x2 = line[2];
            int y2 = line[3];
            cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255), 2);
        }
    }
    return image;
}

std::vector<cv::Vec4i> ImageProcessor::merge_lines(std::vector<cv::Vec4i>& lines, char type) {
    std::vector<cv::Vec4i> merged_lines;
    size_t i = 0;
    if (type == 'v') {
        std::sort(lines.begin(), lines.end(), [](cv::Vec4i& line1, cv::Vec4i& line2) {
            return (line1[1] / 40 < line2[1] / 40) || ((line1[1] / 40 == line2[1] / 40) && (line1[0] < line2[0]));
        });

        while (i < lines.size()) {
            cv::Vec4i line = lines[i];
            size_t next_i = i + 1;
            while (next_i < lines.size()) {
                cv::Vec4i next_line = lines[next_i];
                if (std::abs(line[0] - next_line[0]) < 20) {
                    int x_coordinate = (line[0] + next_line[0]) / 2;
                    cv::Vec4i merged_line = { x_coordinate, std::min(line[1], next_line[1]),
                                               x_coordinate, std::max(line[3], next_line[3]) };
                    merged_lines.push_back(merged_line);
                    i = next_i;
                    next_i++;
                } else {
                    break;
                }
            }
            if (next_i == i + 1) {
                merged_lines.push_back(line);
            }
            i = next_i;
        }
    } else if (type == 'h') {
        std::sort(lines.begin(), lines.end(), [](cv::Vec4i& line1, cv::Vec4i& line2) {
            return (line1[0] / 40 < line2[0] / 40) || ((line1[0] / 40 == line2[0] / 40) && (line1[1] < line2[1]));
        });

        while (i < lines.size()) {
            cv::Vec4i line = lines[i];
            size_t next_i = i + 1;
            while (next_i < lines.size()) {
                cv::Vec4i next_line = lines[next_i];
                if (std::abs(line[1] - next_line[1]) < 20) {
                    int y_coordinate = (line[1] + next_line[1]) / 2;
                    cv::Vec4i merged_line = { std::min(line[0], next_line[0]), y_coordinate,
                                               std::max(line[2], next_line[2]), y_coordinate };
                    merged_lines.push_back(merged_line);
                    i = next_i;
                    next_i++;
                } else {
                    break;
                }
            }
            if (next_i == i + 1) {
                merged_lines.push_back(line);
            }
            i = next_i;
        }
    }
    return merged_lines;
}

double ImageProcessor::average_length( std::vector<cv::Vec4i>& lines) {
    double total_length = 0.0;
    for ( cv::Vec4i& line : lines) {
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];
        double length = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
        total_length += length;
    }

    double average_length = (lines.size() > 0) ? (total_length / lines.size()) : 0.0;
    return average_length;
}

std::vector<cv::Vec4i> ImageProcessor::adjust_line_length( std::vector<cv::Vec4i>& lines, double target_length) {
    std::vector<cv::Vec4i> adjusted_lines;
    for ( cv::Vec4i& line : lines) {
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];
        double length = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
        if (length == 0) {
            adjusted_lines.push_back(line);
        } else {
            double scale_factor = target_length / length;
            // Điều chỉnh tọa độ của điểm cuối của đoạn thẳng, còn điểm đầu giữ nguyên
            int adjusted_x2 = static_cast<int>(x1 + (x2 - x1) * scale_factor);
            int adjusted_y2 = static_cast<int>(y1 + (y2 - y1) * scale_factor);
            cv::Vec4i adjusted_line = { x1, y1, adjusted_x2, adjusted_y2 };
            adjusted_lines.push_back(adjusted_line);
        }
    }
    return adjusted_lines;
}

std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>> ImageProcessor::detect_lines( cv::Mat& edged) {
    std::vector<cv::Vec4i> horizontal_lines;
    std::vector<cv::Vec4i> vertical_lines;

    // cv::Mat blurred = ImageProcessor::blur_image(edged);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edged, lines, 1, CV_PI / 180, 50, 30, 17);

    double max_line_length = 100;

    if (!lines.empty()) {
        for ( auto& line : lines) {
            int x1 = line[0];
            int y1 = line[1];
            int x2 = line[2];
            int y2 = line[3];

            double angle = std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
            double length = std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));

            if ((std::abs(std::fmod(angle, 180.0)) < 20 && ((30 < x1 && x1 < 150 && 30 < x2 && x2 < 150) || (1100 < x1 && x1 < 1190 && 1100 < x2 && x2 < 1190))) && length < max_line_length) {
                int y_coordinate = (y1 + y2) / 2;
                horizontal_lines.push_back({ std::min(x1, x2), y_coordinate, std::max(x1, x2), y_coordinate });
            } else if ((std::abs(std::fmod((angle + 90), 180.0)) < 20 && 200 < x1 && x1 < 1100 && 200 < x2 && x2 < 1100 && 50 < y1 && y1 < 990 && 50 < y2 && y2 < 990) && length < max_line_length) {
                int x_coordinate = (x1 + x2) / 2;
                vertical_lines.push_back({ x_coordinate, std::min(y1, y2), x_coordinate, std::max(y1, y2) });
            }
        }
    }
    horizontal_lines = ImageProcessor::merge_lines(horizontal_lines, 'h');
    vertical_lines = ImageProcessor::merge_lines(vertical_lines, 'v');

    double avg_horizontal_length = ImageProcessor::average_length(horizontal_lines);
    double avg_vertical_length = ImageProcessor::average_length(vertical_lines);

    std::vector<cv::Vec4i> adjusted_horizontal_lines = ImageProcessor::adjust_line_length(horizontal_lines, avg_horizontal_length);
    std::vector<cv::Vec4i> adjusted_vertical_lines = ImageProcessor::adjust_line_length(vertical_lines, avg_vertical_length);

    return { adjusted_horizontal_lines, adjusted_vertical_lines };
}

cv::Mat ImageProcessor::draw_parking_spaces( cv::Mat& image,  std::vector<cv::Vec4i>& lines,  char type,const std::string& filename) {
    cv::Mat image_copy = image.clone();

    if (type == 'h') {
        std::sort(lines.begin(), lines.end(), []( cv::Vec4i& line1,  cv::Vec4i& line2) {
            return (line1[0] / 40 < line2[0] / 40) || ((line1[0] / 40 == line2[0] /40) && (line1[1] < line2[1]));
        });
    } else if (type == 'v') {
        std::sort(lines.begin(), lines.end(), []( cv::Vec4i& line1,  cv::Vec4i& line2) {
            return (line1[1] / 40 < line2[1] / 40) || ((line1[1] / 40 == line2[1] / 40) && (line1[0] < line2[0]));
        });
    }
    std::ofstream outfile;
    if (!filename.empty()) {
        outfile.open(filename,std::ios::app);
    }
    for (size_t i = 0; i < lines.size() - 1; ++i) {
            cv::Vec4i line1 = lines[i];
            cv::Vec4i line2 = lines[i + 1];

            int x_min = std::min({ line1[0], line1[2], line2[0], line2[2] });
            int x_max = std::max({ line1[0], line1[2], line2[0], line2[2] });
            int y_min = std::min({ line1[1], line1[3], line2[1], line2[3] });
            int y_max = std::max({ line1[1], line1[3], line2[1], line2[3] });
            // int x_center = (x_max + x_min) / 2;
            // int y_center = (y_max + y_min) / 2;
            int width = std::abs(x_max - x_min);
            int height = std::abs(y_max - y_min);
            int area = width * height;
            if (2000 < area && area < 6000 && width <100 && height < 100) {
                outfile << x_min << " " << y_min << " " << x_max << " " << y_max << std::endl;
                if (image_copy.channels() == 1){
                    // cv::rectangle(image_copy, cv::Point(x_center-width/2, y_center-height/2), cv::Point(x_center+width/2, y_center+height/2), cv::Scalar(255), 2);
                    cv::rectangle(image_copy, cv::Point(x_min,y_min), cv::Point(x_max,y_max), cv::Scalar(255), 2);
                }
                else{
                    cv::rectangle(image_copy, cv::Point(x_min,y_min), cv::Point(x_max,y_max), cv::Scalar(0, 0, 255), 2);
                }
            }
        }
    outfile.close();

    return image_copy;
}
void ImageProcessor::print_parking_spaces(const std::string& filename){
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Failed to open file!" << filename << std::endl;
        return;
    }
    std::string line;
    int i = 0;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int x_min, y_min, x_max, y_max;
        if (iss >> x_min >> y_min >> x_max >> y_max) {
            std::cout << "Parking space "<< i << " : [" << x_min << ", " << y_min << ", " << x_max << ", " << y_max << "]"<< std::endl;
        }
        ++i;
    }
    infile.close();
}