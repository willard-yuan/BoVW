// 功能函数

#ifndef general_h
#define general_h

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <dirent.h>

std::vector<int> findIntersection(std::vector<int> &a, std::vector<int> &b, std::vector<int> &ia, std::vector<int> &ib) {
    
    std::vector<int> v1 = a;
    std::vector<int> v2 = b;
    
    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());
    
    std::vector<int> v_intersection;
    
    std::set_intersection(v1.begin(), v1.end(),
                          v2.begin(), v2.end(),
                          std::back_inserter(v_intersection));
    std::sort(v_intersection.begin(), v_intersection.end());
    
    for(int i = 0; i < v_intersection.size(); i++){
        
        for(int j = 0; j < a.size(); j++){
            if(v_intersection[i] == a[j]){
                ia.push_back(j+1);
                break;
            }
        }
        
        for(int k = 0; k < b.size(); k++){
            if(v_intersection[i] == b[k]){
                ib.push_back(k+1);
                break;
            }
        }
        
    }
    return v_intersection;
}

// 功能：对给定的向量进行降序排序
// 输入：待降序排序的向量
// 返回值：降序排序后的索引
template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v){
    
    // initialize original index locations
    std::vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    
    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    
    return idx;
}

// 给定"路径+文件名"，获取文件名
void splitFilename(const std::string& str, std::string& fileName){
    unsigned found = (int)str.find_last_of("/\\");
    //path = str.substr(0,found);
    fileName = str.substr(found+1);
    //std::cout << " path: " << str.substr(0,found) << '\n';
    //std::cout << " fileName: " << str.substr(found+1) << '\n';
}

// 四舍五入函数
template <typename T>
T round(T d){
    return floor(d + 0.5);
}

// 获取给定目录下的文件名：文件名
std::vector<std::string> getFilesPath(std::string path = "."){
    DIR*    dir;
    dirent* pdir;
    std::vector<std::string> files;
    dir = opendir(path.c_str());
    while ((pdir = readdir(dir))) {
        if(strcmp(pdir->d_name, ".") && strcmp(pdir->d_name, "..")){
            //std::string tmpPath = path + pdir->d_name;
            //files.push_back(tmpPath);
            std::string tmpPath = pdir->d_name;
            files.push_back(tmpPath);
        }
    }
    return files;
}

// 将STL二维向量转成OpenCV矩阵
template <typename T>
cv::Mat_<T> vec2cvMat_2D(std::vector< std::vector<T> > &inVec){
    int rows = static_cast<int>(inVec.size());
    int cols = static_cast<int>(inVec[0].size());
    
    cv::Mat_<T> resmat(rows, cols);
    for (int i = 0; i < rows; i++){
        resmat.row(i) = cv::Mat(inVec[i]).t();
    }
    return resmat;
}

// 将STL二维向量转成arma矩阵
template <typename T>
arma::mat vec2mat(std::vector<std::vector<T>>&vec){
    int col = (int)vec.size();
    int row = (int)vec[0].size();
    arma::mat A(row, col);
    for(int i = 0; i < col; i++){
        for(int j=0; j < row; j++){
            A(j, i) = vec[i][j];
        }
    }
    return A;
}

// 将特征拼成一位数组
template <typename T>
T * vectors2OneArray(std::vector<std::vector<T>> &descs){
    T * descsToOneArray = (T *)malloc(sizeof(T)*descs.size()*128);
    for(int i = 0; i < descs.size(); i++){
        for(int j = 0; j < 128; j++){
            descsToOneArray[i*128 +j] = descs[i][j];
            //std::cout << std::setiosflags(std::ios::fixed);
            //std::cout << std::setprecision(6) << descsInOne[i*128 +j] << ", ";
        }
        //std::cout << "\n" << std::endl;
    }
    return descsToOneArray;
}

#endif /* general_h */
