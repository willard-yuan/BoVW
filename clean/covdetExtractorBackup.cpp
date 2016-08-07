//
//  covdetExtractor.cpp
//  sift-match-with-ransac
//
//  Created by willard on 1/20/16.
//  Copyright © 2016 wilard. All rights reserved.
//

#include "covdetExtractor.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <iostream>

#include "utils.h"

extern "C" {
#include "vl/covdet.h"
#include "vl/sift.h"
#include "vl/generic.h"
#include "vl/host.h"
#include <time.h>
}

void covdet_keypoints_and_descriptors() {
    /*void covdet_keypoints_and_descriptors(string image_path, bool divide_512,	int verbose, bool display_image, vector<float*>& frames, vector<float*>& descr, unsigned int& number_desc) {*/
    
    // Loading image
    int verbose = 1;
    bool display_image = 1;
    string image_path = "/Users/willard/codes/python/covdet/test.png";
    cv::Mat image = cv::imread(image_path.c_str());   // Read the file
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    
    image.convertTo(image, CV_32FC1);
    
    image = (cv::Mat_<float>)image/255.0;
    
    if (! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find image " << image_path << endl ;
        //return false;
    }
    
    // This is for debugging, checking the image was correctly loaded.
    if (display_image) {
        string window_name = "Image " + image_path;
        namedWindow( window_name, cv::WINDOW_AUTOSIZE );// Create a window for display.
        imshow( window_name, image );               // Show our image inside it.
        cv::waitKey(0);                                 // Wait for a keystroke in the window
    }
    
    typedef enum _VlCovDetDescriptorType{
        VL_COVDET_DESC_SIFT
    } VlCovDetDescriporType;
    
    
    VlCovDetMethod method = VL_COVDET_METHOD_DOG;
    //vl_bool estimateAffineShape = VL_FALSE;
    //vl_bool estimateOrientation = VL_FALSE;
    
    vl_bool doubleImage = VL_TRUE;
    vl_index octaveResolution = -1;
    double edgeThreshold = 10;
    double peakThreshold = 0.01;
    double lapPeakThreshold = -1;
    
    int descriptorType = -1;
    vl_index patchResolution = -1;
    double patchRelativeExtent = -1;
    double patchRelativeSmoothing = -1;
    
    double boundaryMargin = 2.0;
    
    if (descriptorType < 0) descriptorType = VL_COVDET_DESC_SIFT;
    
    switch (descriptorType){
        case VL_COVDET_DESC_SIFT:
            if (patchResolution < 0) patchResolution = 15;
            if (patchRelativeExtent < 0) patchRelativeExtent = 7.5;
            if (patchRelativeSmoothing <0) patchRelativeSmoothing = 1;
            cout << "vl_covdet: patchRelativeExtent " << patchRelativeExtent << endl;
    }
    
    if (image.data) {
        //clock_t t_start = clock();
        // create a detector object: VL_COVDET_METHOD_HESSIAN
        VlCovDet * covdet = vl_covdet_new(method);
        
        // set various parameters (optional)
        vl_covdet_set_first_octave(covdet, doubleImage? -1 : 0);
        
        //vl_covdet_set_octave_resolution(covdet, octaveResolution);
        if (octaveResolution >= 0) vl_covdet_set_octave_resolution(covdet, octaveResolution);
        if (peakThreshold >= 0) vl_covdet_set_peak_threshold(covdet, peakThreshold);
        if (edgeThreshold >= 0) vl_covdet_set_edge_threshold(covdet, edgeThreshold);
        if (lapPeakThreshold >= 0) vl_covdet_set_laplacian_peak_threshold(covdet, lapPeakThreshold);
        
        //vl_covdet_set_target_num_features(covdet, target_num_features);
        //vl_covdet_set_use_adaptive_suppression(covdet, use_adaptive_suppression);
        
        if(verbose){
            std::cout << "vl_covdet: doubling image: " << VL_YESNO(vl_covdet_get_first_octave(covdet) < 0) << ", image size: " << "width: " << image.cols << ", height: " << image.rows << endl;
        }
        
        if (verbose) {
            cout << "vl_covdet: detector: " << vl_enumeration_get_by_value(vlCovdetMethods, method)->name << endl;
            cout << "vl_covdet: peak threshold: " << vl_covdet_get_peak_threshold(covdet) << ", edge threshold: " << vl_covdet_get_edge_threshold(covdet) << endl;
        }
        
        // process the image and run the detector, im.shape(1) is column, im.shape(0) is row
        //see http://www.vlfeat.org/api/covdet_8h.html#affcedda1fdc7ed72d223e0aab003024e for detail
        vl_covdet_put_image(covdet, (float *)image.data, (vl_size)image.cols, (vl_size)image.rows);
        //clock_t t_scalespace = clock();
        vl_covdet_detect(covdet);
        //clock_t t_detect = clock();
        
        if (verbose) {
            vl_size numFeatures = vl_covdet_get_num_features(covdet) ;
            cout << "vl_covdet: " << vl_covdet_get_num_non_extrema_suppressed(covdet) << " features suppressed as duplicate (threshold: "
            << vl_covdet_get_non_extrema_suppression_threshold(covdet) << ")"<< endl;
            cout << "vl_covdet: detected " << numFeatures << " features" << endl;
        }
        
        
        //drop feature on the margin(optimal)
        if(boundaryMargin > 0){
            vl_covdet_drop_features_outside(covdet, boundaryMargin);
            if(verbose){
                vl_size numFeatures = vl_covdet_get_num_features(covdet);
                cout << "vl_covdet: kept " << numFeatures << " inside the boundary margin "<< boundaryMargin << endl;
            }
        }
        
        /* affine adaptation if needed */
        bool estimateAffineShape = true;
        if (estimateAffineShape) {
            if (verbose) {
                vl_size numFeaturesBefore = vl_covdet_get_num_features(covdet) ;
                cout << "vl_covdet: estimating affine shape for " << numFeaturesBefore << " features" << endl;
            }
            
            vl_covdet_extract_affine_shape(covdet) ;
            
            if (verbose) {
                vl_size numFeaturesAfter = vl_covdet_get_num_features(covdet) ;
                cout << "vl_covdet: "<< numFeaturesAfter << " features passed affine adaptation" << endl;
            }
        }
        
        // compute the orientation of the features (optional)
        //clock_t t_affine = clock();
        //vl_covdet_extract_orientations(covdet);
        //clock_t t_orient = clock();
        
        // get feature descriptors
        vl_size numFeatures = vl_covdet_get_num_features(covdet);
        VlCovDetFeature const *feature = (VlCovDetFeature const *)vl_covdet_get_features(covdet);
        VlSiftFilt *sift = vl_sift_new(16, 16, 1, 3, 0);
        vl_index i;
        vl_size dimension = 128;
        vl_size patchSide = 2 * patchResolution + 1;
        
        //std::vector<float> points(6 * numFeatures);
        //std::vector<float> desc(dimension * numFeatures);
        
        std::vector<std::vector<double>> points(numFeatures);
        std::vector<double> desc(dimension);
        std::vector<std::vector<double>> descSet;
        
        std::vector<float> patch(patchSide * patchSide);
        std::vector<float> patchXY(2 * patchSide * patchSide);
        
        double patchStep = (double)patchRelativeExtent / patchResolution;
        
        if (verbose) {
            cout << "vl_covdet: descriptors: type = sift" << ", resolution = " << patchResolution << ", extent = " << patchRelativeExtent << ", smoothing = " << patchRelativeSmoothing << "\n" << endl;
        }
        
        vl_sift_set_magnif(sift, 3.0);
        for (i = 0; i < (signed)numFeatures; ++i) {
            points[i].push_back(feature[i].frame.x);
            points[i].push_back(feature[i].frame.y);
            points[i].push_back(feature[i].frame.a11);
            points[i].push_back(feature[i].frame.a21);
            points[i].push_back(feature[i].frame.a12);
            points[i].push_back(feature[i].frame.a22);
            //std::cout<<setiosflags(ios::fixed);
            //std::cout << setprecision(3) << feature[i].frame.x+1.0 << ", " << feature[i].frame.y+1.0 << ", " << feature[i].frame.a11 << ", "<< feature[i].frame.a21 << ", "<< feature[i].frame.a12 << ", " << feature[i].frame.a22 << ", "<< std::endl;
            
            vl_covdet_extract_patch_for_frame(covdet,
                                              &patch[0],
                                              patchResolution,
                                              patchRelativeExtent,
                                              patchRelativeSmoothing,
                                              feature[i].frame);
            
            vl_imgradient_polar_f(&patchXY[0], &patchXY[1],
                                  2, 2 * patchSide,
                                  &patch[0], patchSide, patchSide, patchSide);
            
            /*vl_sift_calc_raw_descriptor(sift,
             &patchXY[0],
             &desc[dimension * i],
             (int)patchSide, (int)patchSide,
             (double)(patchSide - 1) / 2, (double)(patchSide - 1) / 2,
             (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
             VL_PI / 2);*/
            vl_sift_calc_raw_descriptor(sift,
                                        &patchXY[0],
                                        (vl_sift_pix *)&desc[0],
                                        (int)patchSide, (int)patchSide,
                                        (double)(patchSide - 1) / 2, (double)(patchSide - 1) / 2,
                                        (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
                                        VL_PI / 2);
            descSet.push_back(desc);
        }
        vl_sift_delete(sift);
        vl_covdet_delete(covdet);
        
        //clock_t t_description = clock();
        // std::cout << "t_scalespace " << float(t_scalespace - t_start)/CLOCKS_PER_SEC << "\n";
        // std::cout << "t_detect " << float(t_detect - t_scalespace)/CLOCKS_PER_SEC << "\n";
        // std::cout << "t_affine " << float(t_affine - t_detect)/CLOCKS_PER_SEC << "\n";
        // std::cout << "t_orient " << float(t_orient - t_affine)/CLOCKS_PER_SEC << "\n";
        // std::cout << "description " << float(t_description - t_orient)/CLOCKS_PER_SEC << "\n";
    }
}