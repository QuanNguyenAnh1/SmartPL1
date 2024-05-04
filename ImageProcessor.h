#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <utility>
#include <algorithm>
#include <cmath>
class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();
    cv::Mat preprocessImage( cv::Mat& image);
    std::vector<cv::Point2f> sortPointsClockwise( std::vector<cv::Point2f>& points);
    std::vector<cv::Point2f> selectLargestQuadrilateral( std::vector<cv::Point2f>& points);
    std::vector<cv::Point2f> findQuadrilateralFromPoints( std::vector<cv::Point2f>& points);
    std::vector<cv::Vec4i> detectLines( cv::Mat& edged);
    cv::Mat visualizeLines( cv::Mat& image,  std::vector<cv::Vec4i>& lines);
    std::vector<cv::Point2f> clusterAndValidateIntersections( std::vector<cv::Point2f>& intersections,  cv::Mat& image);
    std::vector<cv::Point2f> lineIntersections( std::vector<cv::Vec4i>& lines,  cv::Mat& image);
    cv::Mat visualizeIntersections( cv::Mat& image,  std::vector<cv::Point2f>& intersections);
    void findQuadrilateralFromPointsAndSave( std::vector<cv::Point2f>& points,  std::string& imageName);
    std::vector<cv::Point2f> processImageAndDetectQuadrilateral( cv::Mat& image);
    cv::Mat drawPoints(cv::Mat image, std::vector<cv::Point2f> points);
    cv::Mat mergeImages( std::vector<cv::Mat>& images);
    std::vector<cv::Mat> readImagesFromFolder( std::string& folderPath);
    cv::Mat blur_image(cv::Mat image);
    cv::Mat visualize_lines(cv::Mat image, std::vector<cv::Vec4i>& lines);
    std::vector<cv::Vec4i> merge_lines(std::vector<cv::Vec4i>& lines,  char type);
    double average_length( std::vector<cv::Vec4i>& lines);
    std::vector<cv::Vec4i> adjust_line_length( std::vector<cv::Vec4i>& lines, double target_length);
    cv::Mat edge_detect(cv::Mat image);
    std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>> detect_lines( cv::Mat& edged);
    cv::Mat draw_parking_spaces( cv::Mat& image,  std::vector<cv::Vec4i>& lines,  char type,const  std::string& file_name = "");
    void print_parking_spaces(const std::string& filename);
};


#endif 
