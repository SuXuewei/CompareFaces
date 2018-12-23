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
    initModule_nonfree();//��ʼ��ģ�飬ʹ��SIFT��SURFʱ�õ�
    Ptr<FeatureDetector> detector = FeatureDetector::create( "SIFT" );//����SIFT���������
    Ptr<DescriptorExtractor> descriptor_extractor = DescriptorExtractor::create( "SIFT" );//������������������
    Ptr<DescriptorMatcher> descriptor_matcher = DescriptorMatcher::create( "BruteForce" );//��������ƥ����
    if( detector.empty() || descriptor_extractor.empty() )
    {
    	cout<<"fail to create detector!";
    }

    //����ͼ��
    Mat img1 = face1;
    Mat img2 = face2;

    //��������
    double t = getTickCount();//��ǰ�δ���
    vector<KeyPoint> keypoints1,keypoints2;
    detector->detect( img1, keypoints1 );//���img1�е�SIFT�����㣬�洢��keypoints1��
    detector->detect( img2, keypoints2 );
    cout<<"ͼ��1���������:"<<keypoints1.size()<<endl;
    cout<<"ͼ��2���������:"<<keypoints2.size()<<endl;

    //����������������������Ӿ��󣬼�������������
    Mat descriptors1,descriptors2;
    descriptor_extractor->compute( img1, keypoints1, descriptors1 );
    descriptor_extractor->compute( img2, keypoints2, descriptors2 );
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout<<"SIFT�㷨��ʱ��"<<t<<"��"<<endl;

    cout<<"ͼ��1�������������С��"<<descriptors1.size()
        <<"����������������"<<descriptors1.rows<<"��ά����"<<descriptors1.cols<<endl;
    cout<<"ͼ��2�������������С��"<<descriptors2.size()
        <<"����������������"<<descriptors2.rows<<"��ά����"<<descriptors2.cols<<endl;

    //����������
    Mat img_keypoints1,img_keypoints2;
    drawKeypoints(img1,keypoints1,img_keypoints1,Scalar::all(-1),0);
    drawKeypoints(img2,keypoints2,img_keypoints2,Scalar::all(-1),0);
//    imshow("Src1",img_keypoints1);
//    imshow("Src2",img_keypoints2);

    //����ƥ��
    vector<DMatch> matches;//ƥ����
    descriptor_matcher->match( descriptors1, descriptors2, matches );//ƥ������ͼ�����������
    cout<<"Match������"<<matches.size()<<endl;

    //����ƥ�����о����������Сֵ
    //������ָ���������������ŷʽ���룬�������������Ĳ��죬ֵԽС��������������Խ�ӽ�
    double max_dist = 0;
    double min_dist = 100;
    for(int i=0; i<matches.size(); i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    cout<<"�����룺"<<max_dist<<endl;
    cout<<"��С���룺"<<min_dist<<endl;

    //ɸѡ���Ϻõ�ƥ���
    vector<DMatch> goodMatches;
    for(int i=0; i<matches.size(); i++)
    {
        if(matches[i].distance < 0.31 * max_dist)
        {
            goodMatches.push_back(matches[i]);
        }
    }

    //����ƥ����
    Mat img_matches;
    //��ɫ���ӵ���ƥ���������ԣ���ɫ��δƥ���������
    drawMatches(img1,keypoints1,img2,keypoints2,goodMatches,img_matches,
                Scalar::all(-1)/*CV_RGB(255,0,0)*/,CV_RGB(0,255,0),Mat(),2);

    imshow("MatchSIFT", img_matches);

    cout<<"goodMatch������"<<goodMatches.size()<<endl;
    if(goodMatches.size() > 15)
    {
    	return 0;
    }

    return 1;
}

//����ԭͼ�ߴ�
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

//�����������
int detectFaceRects(CascadeClassifier &detecter,
		const cv::Mat &frameGray, vector<Rect> &vrectFaces)
{
	detecter.detectMultiScale( frameGray, vrectFaces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
	return 0;
}

int main( int argc, char** argv ){
    std::string resPath = ".\\res\\";

    if( argc != 3){
        cout << "������������" << endl;
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

    //��ȡͼƬ1ԭͼ
	cout << filenameFace1 << endl;
	frameOrgFace1 = imread(filenameFace1);
	if(!frameOrgFace1.data)
	{
		cout << "can not find picture " << filenameFace1 << endl;
		return -1;
	}
    //��ȡͼƬ2ԭͼ
	cout << filenameFace2 << endl;
	frameOrgFace2 = imread(filenameFace2);
	if(!frameOrgFace2.data)
	{
		cout << "can not find picture " << filenameFace2 << endl;
		return -1;
	}

	//��ʾԭͼ
	imshow(WINDOWNAME_FACE1, frameOrgFace1);
	imshow(WINDOWNAME_FACE2, frameOrgFace2);

	//ԭͼ����
	resizeImage(frameOrgFace1, frameFace1);
	resizeImage(frameOrgFace2, frameFace2);

	//ת���ɻҶ�ͼ
	cvtColor(frameFace1, frameGray1, CV_BGR2GRAY);
	cvtColor(frameFace2, frameGray2, CV_BGR2GRAY);
    equalizeHist(frameGray1, frameGray1);  //ֱ��ͼ������
    equalizeHist(frameGray2, frameGray2);  //ֱ��ͼ������

	//����������
	CascadeClassifier ccf;   //��������������
	if (!ccf.load(xmlPath))   //����ѵ���ļ�
	{
		cout << "�޷������ļ�" << xmlPath << endl;
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

    //��ȡ����ͼ��
    Mat roiFace1 = frameGray1(vrectFace1[0]);
    Mat roiFace2 = frameGray2(vrectFace2[0]);
//    imshow(WINDOWNAME_FACE1, roiFace1);
//    imshow(WINDOWNAME_FACE2, roiFace2);

    //����ƥ��
    string result = "����ͬһ����";
    if(0 == matchFaces(roiFace1, roiFace2))
    {
    	result = "��ͬһ����";
    }
    cout << result << endl;

    waitKey(0);
    return 0;
}
