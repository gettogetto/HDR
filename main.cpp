#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>
#include"hdrAlgorithm.h"
using namespace cv;
int main(int argc,char* argv[]){
	hdrAlgorithm hdrathm;

	vector<Mat>multiExposureImages(num);
	string path="SubejctiveTest/13";
	string imageType=".jpg";
	Mat hdrImage;
	hdrathm.getHdrImage(multiExposureImages,path,imageType,hdrImage);
	
	hdrImage.convertTo(hdrImage,CV_8UC3,255.0,0);
	imshow("result",hdrImage);
	//std::cout<<hdrImage.at<uchar>(100,100);
	//imwrite("hdr.jpg",hdrImage);
	waitKey(0);
}