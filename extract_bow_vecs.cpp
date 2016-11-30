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
    
	superluOpts opts; // 几何校正参数
    
    bool verbose = false; // 打印sift提取信息
    bool roosift = true; // sift to rootsift
    
    // Kmeans parameters
    kmeansParameters kmParas;
    kmParas.printMode = true;  // print information when doing kmeans clustering
    kmParas.max_iterations = 5000; // max iterations of kmeans
    
    // Setting word number
    int numWords = 500;
    
    std::string vocWeightSaved = "vocWeights_first1000.mat";
    std::string histsSaved = "hists_first1000.mat";
    
    // search method option: KD-Tree
    std::string annSearch = "OpenCV_KD";
    
    // Obtain the images file names
    std::string imgsRootPath = "/Users/willard/Pictures/first1000/";
    //std::string imgsRootPath = "/Users/willard/codes/cpp/openCVison/bow-beta/bow-beta/images/";
    std::vector<std::string> imgsName = getFilesPath(imgsRootPath);
    int numImgs = (int)imgsName.size();
    
    // Setting sample rate
    int num = numWords*20;
    int numPerImage = ceil((float)num/(float)numImgs);
    
    std::vector<siftDesctor> imgFeatures(numImgs);
    std::vector<std::vector<int>> words(numImgs); // used for reranking later
    
    bowModel BoW(numWords, imgFeatures, words);
    
    std::vector<std::vector<float>> allsifts; // append all sift descs for clustering later
    
    // Extract sift features
    printf("\nstarting sift extraction, image numbers: %d\n", numImgs);
    clock_t begin = clock();
    for(int i = 0; i < numImgs; i++){
        std::string imageFullPath = imgsRootPath + imgsName[i];
        cv::Mat img = cv::imread(imageFullPath.c_str());
        if (!img.data){
            printf("Could not open or find image: %s\n", imageFullPath.c_str());
            break;
         }
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        img.convertTo(img, CV_32FC1); // Convert to float, very important
        
        BoW.imgFeatures[i].imageName = imgsName[i];
        BoW.imgFeatures[i].covdet_keypoints_and_descriptors(img, BoW.imgFeatures[i].frame, BoW.imgFeatures[i].desctor,roosift, verbose);
        
        // Do sample for each image
        arma::vec uniformSubsetIdx = arma::linspace<arma::vec>(0, BoW.imgFeatures[i].desctor.size()-1, numPerImage);
        for(int j = 0; j < uniformSubsetIdx.size(); j++){
            std::vector<float> tmpsift = BoW.imgFeatures[i].desctor[round(uniformSubsetIdx[j])];
            //std::cout << round(uniformSubsetIdx[j]) << std::endl;
            allsifts.push_back(tmpsift);
        }
        printf("extracted %d/%d image, name: %s\n", i , numImgs, BoW.imgFeatures[i].imageName.c_str());
    }
    clock_t end = clock();
    double elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("finished sift extraction, running time: %fs, each image time: %fs\n", elapsed_secs, elapsed_secs/double(numImgs));
    
    // Clustering
    arma::mat sifts_arma_mat = vec2mat(allsifts);
    arma::mat centroids_armaMat;
    printf("\nstarting K-Means, cluster numer: %d, max iterations: %d\n", numWords, (int)kmParas.max_iterations);
    begin = clock();
    arma::kmeans(centroids_armaMat, sifts_arma_mat, BoW.numWords, arma::random_subset, kmParas.max_iterations, kmParas.printMode);
    end = clock();
    elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("finished K-Means, running time: %fs\n", elapsed_secs);

    // Build a KD-Tree
    printf("\nstarting buliding KD tree, Choice: %s\n", annSearch.c_str());
    cv::Mat centroidsOpencvMat(BoW.numWords, 128, CV_64FC1, centroids_armaMat.memptr()); // 转成opencv mat
    centroidsOpencvMat.convertTo(BoW.centroids_opencvMat, CV_32FC1);
    begin = clock();
    cv::flann::Index flann_index = BoW.opencv_buildKDTree(BoW.centroids_opencvMat); // 构建kd树
    end = clock();
    elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("fihished buliding KD tree, running time: %fs\n", elapsed_secs);
    
    // 统计词频
    arma::mat tfMat(numImgs, BoW.numWords, arma::fill::zeros);
    std::vector<int> tmpIdx(BoW.numNeighbors);
    std::vector<float> tmpDis(BoW.numNeighbors);
    printf("\nstarting count the term frequence.............\n");
    begin = clock();
    for(int i = 0; i < numImgs; i++){
        printf("counting %d/%d image TF\n", i , numImgs);
        for(int j = 0; j < BoW.imgFeatures[i].desctor.size(); j++){
            flann_index.knnSearch(BoW.imgFeatures[i].desctor[j], tmpIdx, tmpDis, BoW.numNeighbors,
                                  cv::flann::SearchParams(2000));
            tfMat(i,tmpIdx[0]) = tfMat(i,tmpIdx[0]) + 1;
            BoW.words[i].push_back(tmpIdx[0]);
        }
    }
    end = clock();
    elapsed_secs = double(end - begin)/CLOCKS_PER_SEC;
    printf("fihished counting the TF, running time: %fs\n", elapsed_secs);
    
    // 测试写入操作
    std::ofstream ofs_imgFeatures("imgFeatures.dat", std::ios::binary);
    if (!ofs_imgFeatures) { throw std::runtime_error("Cannot open file."); }
    BoW.Serialize(ofs_imgFeatures);
    ofs_imgFeatures.close();
    printf("\nfinish writing data to file!\n");

    // 计算词频权重
    arma::mat numDocumentsPerWord(1, BoW.numWords);
    for(int i = 0; i < BoW.numWords; i++){
        arma::mat tmpMat = tfMat.col(i);
        numDocumentsPerWord(0,i) = arma::accu(tmpMat != 0);
    }
    arma::mat numDocuments = numImgs*arma::ones(1, BoW.numWords);
    arma::mat vocabWeights = (arma::mat)arma::log(numDocuments/numDocumentsPerWord);
    
    // Save the td-idf weight to file
    vocabWeights.save(vocWeightSaved, arma::raw_ascii);
    
    // 计算直方图
    arma::mat histograms(numImgs, BoW.numWords, arma::fill::zeros);
    for(int i = 0; i < numImgs; i++){
        auto tmpHistogram = tfMat.row(i)%vocabWeights;
        histograms.row(i) = tmpHistogram/std::max(arma::norm(tmpHistogram,2), 1e-12);
    }
    
    // Save the extracted BoW features to file
    histograms.save(histsSaved, arma::raw_ascii);
    
    return 0;
}
