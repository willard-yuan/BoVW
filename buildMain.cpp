/*
 ## Done
  1. 初步测试已有效果，rootSIFT加上去后有了提升
  2. 完成ransac重排
  3. words写入完成
  4. 完成训练和搜索分离
  5. 写入词频权重
 
 ## To do:
 */

#include <math.h>
#include <ctime>

#include "covdetExtractor.hpp"
#include "vl_kdtree.hpp"
#include "general.h"
#include "bow_module.hpp"

extern "C" {
#include "vl/kdtree.h"
}

int main(int argc, char** argv){
    
	superluOpts opts; //几何校正参数
    bool verbose = false; //打印sift提取信息
    bool roosift = true; //提取rootSIFT特征
    
    //kmeans聚类参数
    kmeansParameters kmParas;
    kmParas.printMode = true;
    kmParas.max_iterations = 5000;
    
    // 单词数目
    int numWords = 100;
    
    std::string vocWeightSaved = "vocWeights.mat";
    std::string histsSaved = "hists.mat";
    
    //KD树搜索方法
    std::string annSearch = "OpenCV_KD";
    
    //提取所有图像的特征
    //std::string imgsRootPath = "/Users/willard/Pictures/ukbench/";
    std::string imgsRootPath = "/Users/willard/codes/cpp/openCVison/bow-beta/bow-beta/images/";
    std::vector<std::string> imgsName = getFilesPath(imgsRootPath);
    int imgsNum = (int)imgsName.size();
    
    int num = numWords*20;
    int numPerImage = ceil((float)num/(float)imgsNum); //采样率
    
    std::vector<siftDesctor> imgFeatures(imgsNum);
    std::vector<std::vector<int>> words(imgsNum); //用于后面的重排
    
    bowModel BoW(numWords, imgFeatures, words); //实例化bow
    
    std::vector<std::vector<float>> allsifts; //将所有采样的rootsift特征绑定到一起由于聚类
    
    printf("\nstarting sift extraction, image numbers: %d\n", imgsNum);
    clock_t begin = clock();
    for(int i = 0; i < imgsNum; i++){
        std::string imageFullPath = imgsRootPath + imgsName[i];
        cv::Mat img = cv::imread(imageFullPath.c_str());
        if (!img.data){
            printf("Could not open or find image: %s\n", imageFullPath.c_str());
            break;
         }
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); //转成灰度图像
        img.convertTo(img, CV_32FC1); //转成浮点表示，非常重要
        
        BoW.imgFeatures[i].imageName = imgsName[i];
        BoW.imgFeatures[i].covdet_keypoints_and_descriptors(img, BoW.imgFeatures[i].frame, BoW.imgFeatures[i].desctor,roosift, verbose);
        
        //对每张图片采样，为后面聚类做准备
        arma::vec uniformSubsetIdx = arma::linspace<arma::vec>(0, BoW.imgFeatures[i].desctor.size()-1, numPerImage);
        for(int j = 0; j < uniformSubsetIdx.size(); j++){
            std::vector<float> tmpsift = BoW.imgFeatures[i].desctor[round(uniformSubsetIdx[j])];
            //std::cout << round(uniformSubsetIdx[j]) << std::endl;
            allsifts.push_back(tmpsift);
        }
        printf("extracted %d/%d image, name: %s\n", i , imgsNum, BoW.imgFeatures[i].imageName.c_str());
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("finished sift extraction, running time: %fs, each image time: %fs\n", elapsed_secs, elapsed_secs/double(imgsNum));
    
    //进行聚类
    arma::mat sifts_arma_mat = vec2mat(allsifts);
    arma::mat centroids_armaMat;
    printf("\nstarting K-Means, cluster numer: %d, max iterations: %d\n", numWords, (int)kmParas.max_iterations);
    begin = clock();
    arma::kmeans(centroids_armaMat, sifts_arma_mat, BoW.numWords, arma::random_subset, kmParas.max_iterations, kmParas.printMode);
    end = clock();
    elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("finished K-Means, running time: %fs\n", elapsed_secs);

    //构建kd树
    printf("\nstarting buliding KD tree, Choice: %s\n", annSearch.c_str());
    cv::Mat centroidsOpencvMat(BoW.numWords, 128, CV_64FC1, centroids_armaMat.memptr()); //转成opencv mat
    centroidsOpencvMat.convertTo(BoW.centroids_opencvMat, CV_32FC1);
    begin = clock();
    cv::flann::Index flann_index = BoW.opencv_buildKDTree(BoW.centroids_opencvMat); //构建kd树
    end = clock();
    elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("fihished buliding KD tree, running time: %fs\n", elapsed_secs);
    
    //统计词频
    arma::mat tfMat(imgsNum, BoW.numWords, arma::fill::zeros);
    std::vector<int> tmpIdx(BoW.numNeighbors);
    std::vector<float> tmpDis(BoW.numNeighbors);
    printf("\nstarting count the term frequence.............\n");
    begin = clock();
    for(int i = 0; i < imgsNum; i++){
        printf("counting %d/%d image TF\n", i , imgsNum);
        for(int j = 0; j < BoW.imgFeatures[i].desctor.size(); j++){
            flann_index.knnSearch(BoW.imgFeatures[i].desctor[j], tmpIdx, tmpDis, BoW.numNeighbors, cv::flann::SearchParams(2000));
            tfMat(i,tmpIdx[0]) = tfMat(i,tmpIdx[0]) + 1;
            BoW.words[i].push_back(tmpIdx[0]);
        }
    }
    end = clock();
    elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("fihished counting the TF, running time: %fs\n", elapsed_secs);
    
    //测试写入操作
    std::ofstream ofs_imgFeatures("imgFeatures.dat", std::ios::binary);
    if (!ofs_imgFeatures) { throw std::runtime_error("Cannot open file."); }
    BoW.Serialize(ofs_imgFeatures);
    ofs_imgFeatures.close();
    printf("\nfinish writing data to file!\n");

    //计算词频权重
    arma::mat numDocumentsPerWord(1, BoW.numWords);
    for(int i = 0; i < BoW.numWords; i++){
        arma::mat tmpMat = tfMat.col(i);
        numDocumentsPerWord(0,i) = arma::accu(tmpMat != 0);
    }
    arma::mat numDocuments = imgsNum*arma::ones(1, BoW.numWords);
    arma::mat vocabWeights = (arma::mat)arma::log(numDocuments/numDocumentsPerWord);
    vocabWeights.save(vocWeightSaved, arma::raw_ascii); //保存词频权重
    
    //计算直方图
    arma::mat histograms(imgsNum, BoW.numWords, arma::fill::zeros);
    for(int i = 0; i < imgsNum; i++){
        auto tmpHistogram = tfMat.row(i)%vocabWeights;
        histograms.row(i) = tmpHistogram/std::max(arma::norm(tmpHistogram,2), 1e-12);
    }
    histograms.save(histsSaved, arma::raw_ascii); //保存词频直方图

    //测试查询
    /*int queryID = 8;
    arma::mat queryhistogram = histograms.row(queryID); //查询图像
    std::vector<float> scores;
    for(int i = 0; i < imgsNum; i++){
        float tmpscore = arma::dot(queryhistogram.t(), histograms.row(i)); // cosine distance
        scores.push_back(tmpscore);
    }
 
    //相似度排序
    std::vector<size_t> sortedIdx = sort_indexes(scores);
    for (auto idx: sortedIdx) {
        printf("%s, score: %4f\n", BoW.imgFeatures[idx].imageName.c_str(), scores[idx]);
    }*/
    
    return 0;
}
