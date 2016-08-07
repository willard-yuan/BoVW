#include "bow_module.hpp"

cv::flann::Index bowModel::opencv_buildKDTree(cv::Mat &centroids_opencvMat){
    cv::flann::Index flann_index(centroids_opencvMat, cv::flann::KDTreeIndexParams(512), cvflann::FLANN_DIST_EUCLIDEAN);
    flann_index.save("myFirstIndex"); //保存索引结构
    cv::flann::Index flann;
    flann.load(centroids_opencvMat,"myFirstIndex");
    flann.save("mySecondIndex");
    return flann_index;
}

void saveVocWeights(){};
void saveHists(){};

/*std::vector<float> reRanking(bowModel BoW, std::vector<float> scores, int queryID, int reRankingDepth, std::vector<size_t> sortedIdx, superluOpts opts){
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
        
        if((1)){
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
    return scores;
}*/