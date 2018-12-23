//============================================================================
// Name        : CompareFaces.cpp
// Author      : g_su_xw
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

const string WINDOWNAME_FACE1 = "Face Detection Face 1";
const string WINDOWNAME_FACE2 = "Face Detection Face 2";

int matchFaces(const Mat &face1, const Mat &face2)
{
	return 1;
//    cv::initModule_nonfree();
//    Mat img1_mat = face1;
//    Mat img2_mat = face2;
//
//    std::vector<KeyPoint> img1_keypoint,img2_keypoint;
//    /*特征点检测*/
//    //  FeatureDetector *sift_detector = FeatureDetector::create("SIFT");
//    Ptr<FeatureDetector> sift_detector = FeatureDetector::create("SIFT");
//    sift_detector->detect(img1_mat,img1_keypoint);
//    sift_detector->detect(img2_mat,img2_keypoint);
//    /*在图片上显示特征点*/
//    Mat img_kp_1,img_kp_2;
//    drawKeypoints(img1_mat,img1_keypoint,img_kp_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//    drawKeypoints(img2_mat,img2_keypoint,img_kp_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//    imshow("sift_keypoint_image 1",img_kp_1);
//    imshow("sift_keypoint_image 2",img_kp_2);
//    /*特征向量提取*/
//    Mat img1_descriptor,img2_descriptor;
//    Ptr<DescriptorExtractor> sift_descriptor_extractor = DescriptorExtractor::create("SIFT");
//    sift_descriptor_extractor->compute(img1_mat,img1_keypoint,img1_descriptor);
//    sift_descriptor_extractor->compute(img2_mat,img2_keypoint,img2_descriptor);
//    /*特征点匹配*/
//    std::vector<DMatch> img_match;
//    Ptr<DescriptorMatcher> bruteforce_matcher = DescriptorMatcher::create("BruteForce");
//    bruteforce_matcher->match(img1_descriptor,img2_descriptor,img_match);
//
//    /*在图片上显示匹配结果*/
//    Mat match_img;
//    drawMatches(img1_mat,img1_keypoint,img2_mat,img2_keypoint,accurate_match,match_img);
//    imshow("match image",match_img);
//    waitKey(0);
}

//处理原图尺寸
int resizeImage(const cv::Mat &srcMat, cv::Mat &destMat)
{
	if((srcMat.cols > 800) || (srcMat.rows > 600))
	{
		resize(srcMat, destMat,
				Size(srcMat.cols / 2, srcMat.rows / 2),
				0, 0, INTER_LINEAR);
	} else {
		destMat = srcMat;
	}

	return 0;
}

int main( int argc, char** argv ){
    std::string resPath = "D:\\eclipse-workspace\\MinGWDebugTest\\res\\";
    //std::string resPath = ".\\res\\";

	string xmlPath = resPath + "haarcascade_frontalface_alt.xml";
	CascadeClassifier ccf;   //创建分类器对象
	if (!ccf.load(xmlPath))   //加载训练文件
	{
		cout << "不能加载文件" << xmlPath << endl;
		return 0;
	}

    Mat frameFace1;
    Mat frameFace2;
    Mat frameOrgFace1;
    Mat frameOrgFace2;
    Mat frameGray1;
    Mat frameGray2;
    vector<Rect> vrectFace1;
    vector<Rect> vrectFace2;
    string filenameFace1 = resPath + "pic17.jpg";
    string filenameFace2 = resPath + "pic18.jpg";

    namedWindow(WINDOWNAME_FACE1);
    namedWindow(WINDOWNAME_FACE2);

    //读取图片1原图
	cout << filenameFace1 << endl;
	frameOrgFace1 = imread(filenameFace1);
	if(!frameOrgFace1.data)
	{
		cout << "can not find picture " << filenameFace1 << endl;
		return -1;
	}
    //读取图片2原图
	cout << filenameFace2 << endl;
	frameOrgFace2 = imread(filenameFace2);
	if(!frameOrgFace2.data)
	{
		cout << "can not find picture " << filenameFace2 << endl;
		return -1;
	}

	//显示原图
	imshow(WINDOWNAME_FACE1, frameOrgFace1);
	imshow(WINDOWNAME_FACE2, frameOrgFace2);

	//原图缩放
	resizeImage(frameOrgFace1, frameFace1);
	resizeImage(frameOrgFace2, frameFace2);

	//转换成灰度图
	cvtColor(frameFace1, frameGray1, CV_BGR2GRAY);
	cvtColor(frameFace2, frameGray2, CV_BGR2GRAY);
    equalizeHist(frameGray1, frameGray1);  //直方图均衡行
    equalizeHist(frameGray2, frameGray2);  //直方图均衡行

	//人脸区域检测
    vrectFace1.clear();
    vrectFace2.clear();
    ccf.detectMultiScale( frameGray1, vrectFace1, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    if(vrectFace1.size())
    {
    	rectangle(frameFace1, vrectFace1[0], Scalar(0,255,0));
    }
    imshow(WINDOWNAME_FACE1, frameFace1);
    ccf.detectMultiScale( frameGray2, vrectFace2, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    if(vrectFace2.size())
    {
    	rectangle(frameFace2, vrectFace2[0], Scalar(0,255,0));
    }
    imshow(WINDOWNAME_FACE2, frameFace2);

    //截取人脸图像
    Mat roiFace1 = frameGray1(vrectFace1[0]);
    Mat roiFace2 = frameGray2(vrectFace2[0]);
    imshow(WINDOWNAME_FACE1, roiFace1);
    imshow(WINDOWNAME_FACE2, roiFace2);

    //人脸匹配
    string result = "不是同一个人";
    if(0 == matchFaces(roiFace1, roiFace2))
    {
    	result = "是同一个人";
    }
    cout << result << endl;

    waitKey(0);
    return 0;
}
