/*  First: Created by willard on 1/20/16.

 ## To improve:
     1. 写入文件名的时候，采用string的写法为什么不行
 */


#ifndef covdetExtractor_hpp
#define covdetExtractor_hpp

#include "utils.h"


class siftDesctor{
public:
    siftDesctor(){};
    std::string imageName;
    std::vector<std::vector<float>> frame;
    std::vector<std::vector<float>> desctor;
    void covdet_keypoints_and_descriptors(cv::Mat &img, std::vector<std::vector<float>> &frames, std::vector<std::vector<float>> &desctor, bool rooSIFT, bool verbose);
    void flip_descriptor(std::vector<float> &dst, float *src);
    std::vector<float> rootsift(std::vector<float> &dst);
    
    void Serialize(std::ofstream &outfile) const {
        std::string tmpImageName = imageName;
        int strSize = (int)imageName.size();
        outfile.write((char *)&strSize, sizeof(int));
        outfile.write((char *)&tmpImageName[0], sizeof(char)*strSize); // 写入文件名
        //outfile.write(tmpImageName.c_str(), sizeof(char)*strSize); // 这种写法也ok
        
        int descSize = (int)desctor.size();
        outfile.write((char *)&descSize, sizeof(int));
        
        // 写入sift特征
        for(int i = 0; i < descSize; i++ ){
            outfile.write((char *)&(desctor[i][0]), sizeof(float) * 128);
            outfile.write((char *)&(frame[i][0]), sizeof(float) * 6);
        }
        
    }
    
    static siftDesctor Deserialize(std::ifstream &ifs) {
        siftDesctor siftDesc;
        int strSize = 0;
        ifs.read((char *)&strSize, sizeof(int)); // 写入文件名
        siftDesc.imageName = "";
        siftDesc.imageName.resize(strSize);
        ifs.read((char *)&(siftDesc.imageName[0]), sizeof(char)*strSize); // 读入文件名
        
        int descSize = 0;
        ifs.read((char *)&descSize, sizeof(int));
        
        // 写入sift特征和frame
        for(int i = 0; i < descSize; i++ ){
            std::vector<float> tmpDesc(128);
            ifs.read((char *)&(tmpDesc[0]), sizeof(float) * 128);
            siftDesc.desctor.push_back(tmpDesc);
            
            std::vector<float> tmpFrame(6);
            ifs.read((char *)&(tmpFrame[0]), sizeof(float) * 6);
            siftDesc.frame.push_back(tmpFrame);
        }
        
        return siftDesc;
    }
    
};

#endif /* covdetExtractor_hpp */
