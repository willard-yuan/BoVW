/*
 ## 测试
  1. 初步测试已有效果，rootSIFT加上去后有了提升
  2. 完成ransac重排
  3. 解决数据读写保存问题
 */

#include "covdetExtractor.hpp"
#include "vl_kdtree.hpp"
#include "general.h"
#include "bow_module.hpp"
#include "utils.h"

#include <math.h>

extern "C" {
#include "vl/kdtree.h"
}

int main(int argc, char** argv){
    
    std::string vocWeightSaved = "vocWeights.mat";
    std::string histsSaved = "hists.mat";
    int showNum = 20;
    
	superluOpts opts; //几何校正参数
    
    //提取所有图像的特征
    std::string imgsRootPath = "/Users/willard/codes/cpp/openCVison/bow-beta/bow-beta/images/";
    std::vector<std::string> imgsPath = getFilesPath(imgsRootPath);
    int imgsNum = (int)imgsPath.size();
    
    arma::mat vocabWeights;
    vocabWeights.load(vocWeightSaved, arma::raw_ascii);
    
    arma::mat histograms;
    histograms.load(histsSaved, arma::raw_ascii);
    
    //测试查询
    // Need to improve: 用矩阵代替for循环
    int queryID = 4;
    arma::mat queryhistogram = histograms.row(queryID); //查询图像
    std::vector<float> scores;
    for(int i = 0; i < imgsNum; i++){
        float tmpscore = arma::dot(queryhistogram.t(), histograms.row(i)); // 计算余弦距离
        scores.push_back(tmpscore);
    }
    
    //输出分数
    /*std::cout << std::setiosflags(std::ios::fixed);
    for(auto score: scores){
        std::cout << std::setprecision(6) << score << " ";
    }
    std::cout << "\n" << std::endl;*/
    
    std::vector<cv::Mat> vecMat;
    
    //相似度排序
    std::vector<size_t> sortedIdx = sort_indexes(scores);
    for(int i = 0; i < showNum; i++){
    //for (auto idx: sortedIdx) {
        std::cout << imgsPath.at(sortedIdx[i]) << ": " << scores[sortedIdx[i]] << std::endl;
        cv::Mat tmpImg = cv::imread(imgsRootPath+imgsPath.at(sortedIdx[i]), CV_8UC3);
        vecMat.push_back(tmpImg);
        
    }
    
    std::cout << "\n" << std::endl;
    
    cv::Mat combinedImgs =  makeCanvas(vecMat, 800, 5);
    cv::imshow("query results", combinedImgs);
    cv::waitKey();
    
    return 0;
}
