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
	imshow("1",hdrImage);
	waitKey(0);
}