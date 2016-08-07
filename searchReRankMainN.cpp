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

#include <math.h>

extern "C" {
#include "vl/kdtree.h"
}

int main(int argc, char** argv){
    
    std::string vocWeightSaved = "vocWeights.mat";
    std::string histsSaved = "hists.mat";
    int showNum = 10;
    
	superluOpts opts; //几何校正参数
    
    //提取所有图像的特征
    std::string imgsRootPath = "/Users/willard/codes/cpp/openCVison/bow-alpha/bow-alpha/images/";
    std::vector<std::string> imgsPath = getFilesPath(imgsRootPath);
    int imgsNum = (int)imgsPath.size();
    
    
    // 测试读入操作
    std::ifstream ifs_imgFeatures("imgFeatures.dat", std::ios::binary);
    if (!ifs_imgFeatures) { throw std::runtime_error("Cannot open file."); }
    bowModel BoW = bowModel::Deserialize(ifs_imgFeatures);
    ifs_imgFeatures.close();
    
    
    arma::mat vocabWeights;
    vocabWeights.load(vocWeightSaved, arma::raw_ascii);
    
    arma::mat histograms;
    histograms.load(histsSaved, arma::raw_ascii);
    
    //测试查询
    // Need to improve: 用矩阵代替for循环
    int queryID = 2;
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
    
    //相似度排序
    std::vector<size_t> sortedIdx = sort_indexes(scores);
    for(int i = 0; i < showNum; i++){
    //for (auto idx: sortedIdx) {
        std::cout << sortedIdx[i] << ": " << scores[sortedIdx[i]] << std::endl;
    }
    
     std::cout << "\n" << std::endl;
    
    //重排
    int reRankingDepth = 10;
    std::vector<int> queryWords = BoW.words[queryID];
    for(int i = 0; i < reRankingDepth; i++){
        std::vector<int> ia;
        std::vector<int> ib;
        std::vector<std::vector<int>> matchedIdx(2);
        std::vector<int> intersect = findIntersection(queryWords, BoW.words[sortedIdx[i]], ia, ib);
        matchedIdx[0] = ia;
        matchedIdx[1] = ib;
        
        arma::mat frames1 = vec2mat(BoW.imgFeatures[queryID].frame);
        arma::mat frames2 = vec2mat(BoW.imgFeatures[sortedIdx[i]].frame);
        arma::mat matches = vec2mat(matchedIdx);
        matches = matches.t();
        
        arma::uvec inliers_final = geometricVerification(frames1, frames2, matches, opts);
        
        if(inliers_final.size() >= 6){
            scores[sortedIdx[i]] = scores[sortedIdx[i]] + inliers_final.size();
        }
        
        if(/* DISABLES CODE */ (1)){
            std::vector<cv::Point2f> srcPoints, dstPoints;
            arma::mat matches_geo = matches.cols(inliers_final);
            //cout << matches_geo.n_rows << "+++++" <<matches_geo.n_cols << endl;
            for (unsigned int i = 0; i < matches_geo.n_cols; ++i){
                cv::Point2f pt1, pt2;
                //cout << matches_geo.at(0, i) << " " << matches_geo.at(1, i) << endl;
                pt1.x = frames1.at(0, matches_geo.at(0, i) - 1);
                pt1.y = frames1.at(1, matches_geo.at(0, i) - 1);
                pt2.x = frames2.at(0, matches_geo.at(1, i) - 1);
                pt2.y = frames2.at(1, matches_geo.at(1, i) - 1);
                srcPoints.push_back(pt1);
                dstPoints.push_back(pt2);
            }
            
            cv::Mat QImage = cv::imread(BoW.imgFeatures[queryID].imageName);
            cv::Mat DImage = cv::imread(BoW.imgFeatures[sortedIdx[i]].imageName);
                plotMatches(QImage, DImage, srcPoints, dstPoints);
        }
    }
    
    //重排相似度排序
    std::vector<size_t> resortedIdx = sort_indexes(scores);
    for (auto idx: resortedIdx) {
        std::cout << idx << ": " << scores[idx] << std::endl;
    }
    
    return 0;
}
