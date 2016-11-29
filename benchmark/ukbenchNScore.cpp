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
#include <ctime>

extern "C" {
#include "vl/kdtree.h"
}

int main(int argc, char** argv){
    
    std::string vocWeightSaved = "vocWeights.mat";
    std::string histsSaved = "hists.mat";
    int showNum = 4;
    
	superluOpts opts; //几何校正参数
    
    //提取所有图像的特征
    std::string imgsRootPath = "/Users/willard/Pictures/ukbench/";
    std::vector<std::string> imgsPath = getFilesPath(imgsRootPath);
    int imgsNum = (int)imgsPath.size();
    
    arma::mat vocabWeights;
    vocabWeights.load(vocWeightSaved, arma::raw_ascii);
    
    arma::mat histograms;
    histograms.load(histsSaved, arma::raw_ascii);
    
    //测试查询
    // Need to improve: 用矩阵代替for循环
    clock_t begin = clock();
    float sumScores = 0;
    for(int k = 0; k < imgsPath.size(); k++){
        int queryID = std::atoi(imgsPath[k].substr(7,5).c_str());
        arma::mat queryhistogram = histograms.row(queryID); //查询图像
        std::vector<float> scores;
        for(int i = 0; i < imgsNum; i++){
            float tmpscore = arma::dot(queryhistogram.t(), histograms.row(i)); // 计算余弦距离
            scores.push_back(tmpscore);
        }
        
        //相似度排序
        std::vector<size_t> sortedIdx = sort_indexes(scores);
        for(int i = 0; i < showNum; i++){
            printf("%s, score: %f\n", imgsPath[sortedIdx[i]].c_str(), scores[sortedIdx[i]]);
        }
            
        float score = 0;
        for (int i = 0; i < 4; i++) {
            if (std::floor(sortedIdx[i]/4.0) == floor(queryID/4)) ++ score;
        }
        
        printf("image(%d/%d): %s, score: %f\n\n", k+1, (int)imgsPath.size(), imgsPath[k].c_str(),score);
        
        sumScores = sumScores + score;
    }
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    
    printf("NS-Score: %f, search time: %f\n", sumScores/imgsPath.size(), elapsed_secs/imgsPath.size());
    
    return 0;
}
