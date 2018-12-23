//============================================================================
// Name        : CompareFaces.cpp
// Author      : g_su_xw
// Version     :
// Copyright   : Your copyright notice
// Description : Ansi-style
//============================================================================

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <iostream>

using namespace std;
using namespace cv;

const string WINDOWNAME_FACE1 = "Face Detection Face 1";
const string WINDOWNAME_FACE2 = "Face Detection Face 2";

int matchFaces(const Mat &face1, const Mat &face2)
{
    initModule_nonfree();//初始化模块，使用SIFT或SURF时用到
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );//创建SIFT特征检测器
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );//创建特征向量生成器
    Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//创建特征匹配器
    if( detector.empty() || descriptor_extractor.empty() )
    {
    	cout<<"fail to create detector!";
    }

    //读入图像
    Mat img1 = face1;
    Mat img2 = face2;

    //特征点检测
    double t = getTickCount();//当前滴答数
    vector<KeyPoint> keypoints1,keypoints2;
    detector->detect( img1, keypoints1 );//检测img1中的SIFT特征点，存储到keypoints1中
    detector->detect( img2, keypoints2 );
    cout<<"图像1特征点个数:"<<keypoints1.size()<<endl;
    cout<<"图像2特征点个数:"<<keypoints2.size()<<endl;

    //根据特征点计算特征描述子矩阵，即特征向量矩阵
    Mat descriptors1,descriptors2;
    descriptor_extractor->compute( img1, keypoints1, descriptors1 );
    descriptor_extractor->compute( img2, keypoints2, descriptors2 );
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout<<"SIFT算法用时："<<t<<"秒"<<endl;

    cout<<"图像1特征描述矩阵大小："<<descriptors1.size()
        <<"，特征向量个数："<<descriptors1.rows<<"，维数："<<descriptors1.cols<<endl;
    cout<<"图像2特征描述矩阵大小："<<descriptors2.size()
        <<"，特征向量个数："<<descriptors2.rows<<"，维数："<<descriptors2.cols<<endl;

    //画出特征点
    Mat img_keypoints1,img_keypoints2;
    drawKeypoints(img1,keypoints1,img_keypoints1,Scalar::all(-1),0);
    drawKeypoints(img2,keypoints2,img_keypoints2,Scalar::all(-1),0);
//    imshow("Src1",img_keypoints1);
//    imshow("Src2",img_keypoints2);

    //特征匹配
    vector<DMatch> matches;//匹配结果
    descriptor_matcher->match( descriptors1, descriptors2, matches );//匹配两个图像的特征矩阵
    cout<<"Match个数："<<matches.size()<<endl;

    //计算匹配结果中距离的最大和最小值
    //距离是指两个特征向量间的欧式距离，表明两个特征的差异，值越小表明两个特征点越接近
    double max_dist = 0;
    double min_dist = 100;
    for(int i=0; i<matches.size(); i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout<<"最大距离："<<max_dist<<endl;
    cout<<"最小距离："<<min_dist<<endl;

    //筛选出较好的匹配点
    vector<DMatch> goodMatches;
    for(int i=0; i<matches.size(); i++)
    {
        if(matches[i].distance < 0.31 * max_dist)
        {
            goodMatches.push_back(matches[i]);
        }
    }

    //画出匹配结果
    Mat img_matches;
    //红色连接的是匹配的特征点对，绿色是未匹配的特征点
    drawMatches(img1,keypoints1,img2,keypoints2,goodMatches,img_matches,
                Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);

    imshow("MatchSIFT", img_matches);

    cout<<"goodMatch个数："<<goodMatches.size()<<endl;
    if(goodMatches.size() > 15)
    {
    	return 0;
    }

    return 1;
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

//检测人脸区域
int detectFaceRects(CascadeClassifier &detecter,
		const cv::Mat &frameGray, vector<Rect> &vrectFaces)
{
	detecter.detectMultiScale( frameGray, vrectFaces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	return 0;
}

int main( int argc, char** argv ){
    std::string resPath = ".\\res\\";

    if( argc != 3){
        cout << "参数个数错误" << endl;
        return -1;
    }

    string filenameFace1 = resPath + argv[1] + ".jpg";
    string filenameFace2 = resPath + argv[2] + ".jpg";
	string xmlPath = resPath + "haarcascade_frontalface_alt.xml";

    Mat frameFace1;
    Mat frameFace2;
    Mat frameOrgFace1;
    Mat frameOrgFace2;
    Mat frameGray1;
    Mat frameGray2;
    vector<Rect> vrectFace1;
    vector<Rect> vrectFace2;

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
	CascadeClassifier ccf;   //创建分类器对象
	if (!ccf.load(xmlPath))   //加载训练文件
	{
		cout << "无法加载文件" << xmlPath << endl;
		return 0;
	}
    vrectFace1.clear();
    vrectFace2.clear();
    detectFaceRects(ccf, frameGray1, vrectFace1);
    if(vrectFace1.size())
    {
    	rectangle(frameFace1, vrectFace1[0], Scalar(0,255,0));
    }
    imshow(WINDOWNAME_FACE1, frameFace1);
    detectFaceRects(ccf, frameGray2, vrectFace2);
    if(vrectFace2.size())
    {
    	rectangle(frameFace2, vrectFace2[0], Scalar(0,255,0));
    }
    imshow(WINDOWNAME_FACE2, frameFace2);

    //截取人脸图像
    Mat roiFace1 = frameGray1(vrectFace1[0]);
    Mat roiFace2 = frameGray2(vrectFace2[0]);
//    imshow(WINDOWNAME_FACE1, roiFace1);
//    imshow(WINDOWNAME_FACE2, roiFace2);

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
