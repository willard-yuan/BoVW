#ifndef bow_module_hpp
#define bow_module_hpp

#include <stdio.h>
#include "utils.h"
#include "covdetExtractor.hpp"

struct kmeansParameters{
    bool            printMode;  // 是否打印输出聚类的信息
    arma::uword     max_iterations ;  // 设置聚类最大迭代次数
};

class bowModel {
public:
    bowModel(){};
    bowModel(int _numWords,std::vector<siftDesctor> _imgFeatures, std::vector<std::vector<int>> _words):numWords(_numWords),imgFeatures(_imgFeatures),words(_words){};
    
    int numNeighbors = 1;
    int numWords;
    std::vector<siftDesctor> imgFeatures;
    std::vector<std::vector<int>> words;
    cv::Mat centroids_opencvMat;
    
    cv::flann::Index opencv_buildKDTree(cv::Mat &centroids_opencvMat);
    
    void saveVocWeights();
    void saveHists();
    
    void Serialize(std::ofstream &outfile) const {
        int imgFeatsSize = (int)imgFeatures.size();
        outfile.write((char *)&imgFeatsSize, sizeof(int));
        // 写入imgFeatures和words
        for(int i = 0; i < imgFeatsSize; i++ ){
            imgFeatures[i].Serialize(outfile);
            outfile.write((char *)&(words[i][0]), sizeof(int) * imgFeatures[i].desctor.size());
        }
        
    }
    
    static bowModel Deserialize(std::ifstream &ifs) {
        bowModel BoW;
        int imgFeatsSize;
        ifs.read((char *)&imgFeatsSize, sizeof(int));
        
        BoW.words.resize(imgFeatsSize);
        
        for (int i = 0; i < imgFeatsSize; i++) {
            // 读入特征
            auto siftDesc = siftDesctor::Deserialize(ifs);
            BoW.imgFeatures.push_back(siftDesc);
            // 读入word
            BoW.words[i].resize(siftDesc.desctor.size());
            ifs.read((char *)&(BoW.words[i][0]), sizeof(int) * siftDesc.desctor.size());
        }
        return BoW;
    }
    
};

//std::vector<float> reRanking(bowModel BoW, std::vector<float> scores, int queryID, int reRankingDepth, std::vector<size_t> sortedIdx, superluOpts opts);

#endif /* bow_module_hpp */
