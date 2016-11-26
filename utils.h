/*
 * FileName : utils.c
 * Author   : yongyuanstu@gmail.com
 * Version  : v1.0
 * Date     : 29 Aug 2015 08:31:41 PM CST
 * Brief    :
 */

#ifndef utils_h
#define utils_h

#include <iostream>
#include <time.h>
#include <vector>

#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/flann/flann.hpp>

#include <fstream>

#include "armadillo"

float * descsInOne(std::vector<std::vector<float>> &descs);

struct superluOpts{
	int             tolerance1 = 20;  // default: true
	int             tolerance2 = 15;    // default: false
	int             tolerance3 = 8; // default: 1.0
	unsigned int    minInliers = 6;
	unsigned int    numRefinementIterations = 8;
};

//生成随机颜色
static cv::Scalar randomColor(cv::RNG& rng);

//画匹配点
void plotMatches(const cv::Mat &src, const cv::Mat &obj, std::vector<cv::Point2f> &srcPoints, std::vector<cv::Point2f> &dstPoints);

//使用OpenCV自带的寻找内点
void findInliers(std::vector<cv::KeyPoint> &qKeypoints, std::vector<cv::KeyPoint> &objKeypoints, std::vector<cv::DMatch> &matches, const std::string &imgfn, const std::string &objFileName);

//自己写的寻找内点
arma::mat centering(arma::mat &x);
arma::mat toAffinity(arma::mat &f);
arma::uvec geometricVerification(const arma::mat &frames1, const arma::mat &frames2, const arma::mat &matches, const superluOpts &opts);

//词袋模型重排
/*std::vector<int> findIntersection(std::vector<int> &a, std::vector<int> &b, std::vector<int> &ia, std::vector<int> &ib);

template <typename T>
cv::Mat_<T> vec2cvMat_2D(std::vector< std::vector<T> > &inVec);

template <typename T>
arma::mat vec2mat(std::vector<std::vector<T>>&vec);*/


cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows);

#endif /* utils_h */