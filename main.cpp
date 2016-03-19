#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>
#include<iostream>
using namespace std;
using namespace cv;
#define num 6
Mat getContrastImage(const Mat& srcImage){
	Mat dstImage;
	Mat kenel(3,3,CV_32F,Scalar::all(0));
	kenel.at<float>(0,1)=-1;
	kenel.at<float>(1,0)=-1;
	kenel.at<float>(1,1)= 9;
	kenel.at<float>(1,2)=-1;
	kenel.at<float>(2,1)=-1;
	filter2D(srcImage,dstImage,srcImage.depth(),kenel);
	return dstImage;
}
void getMultiContrastImage(vector<Mat>&multiExposureImages,vector<Mat>&multiContrastImages){
	for(int i=0;i<multiExposureImages.size();i++){
		Mat grayImage;
		cvtColor(multiExposureImages[i],grayImage,CV_BGR2GRAY);
		multiContrastImages[i]=getContrastImage(grayImage);
	}
}
void getWeightMapImage_Contrast(vector<Mat>& multiContrast,vector<Mat>& multiMapImage){
	int n=multiContrast.size();
	Size size=multiContrast[0].size();
	int w=size.width;
	int h=size.height;
	for(int n1=0;n1<n;n1++){
		for(int i=0;i<h;i++){
			for(int j=0;j<w;j++){
				float sum=0;
				for(int n2=0;n2<n;n2++){
					sum+=(  multiContrast[n2].at<uchar>(i,j)   );
				}
				multiMapImage[n1].at<float>(i,j)=multiContrast[n1].at<uchar>(i,j)/sum;
			}
		}		
	}
}

Mat getExposureFusionImage(vector<Mat>&multiExposureImages,vector<Mat>&multiWeightMapImage){

	Size size=multiExposureImages[0].size();
	int h=size.height;
	int w=size.width;
	Mat exposureFusionImage(h,w,CV_32FC3,Scalar::all(0));
	for(std::size_t i=0;i<multiExposureImages.size();i++){
		vector<Mat>rgbPlanes;
		cv::split(multiExposureImages[i],rgbPlanes);
		//为了Mat之间的运算，先转变成一致类型
		rgbPlanes[0].convertTo(rgbPlanes[0],CV_32FC1,1.0/255);
		rgbPlanes[1].convertTo(rgbPlanes[1],CV_32FC1,1.0/255);
		rgbPlanes[2].convertTo(rgbPlanes[2],CV_32FC1,1.0/255);

		rgbPlanes[0]=rgbPlanes[0].mul(multiWeightMapImage[i]);
		rgbPlanes[1]=rgbPlanes[1].mul(multiWeightMapImage[i]);
		rgbPlanes[2]=rgbPlanes[2].mul(multiWeightMapImage[i]);

		Mat rgbMergeImage;
		cv::merge(rgbPlanes,rgbMergeImage);
		exposureFusionImage+=rgbMergeImage;
	}
	return exposureFusionImage;
}

void loadMultiExposureImages(vector<Mat>&multiExposureImages,const string& path,const string& imageType){
	for(std::size_t i=0;i<multiExposureImages.size();i++){
		multiExposureImages[i]=imread(path+"/"+to_string(long long(i+1))+imageType);
	}
}
/////////////////////////////////////////////////////
void getVchannelOfHSV(const vector<Mat>& multiExposureImages,vector<Mat>& outputImages){
	size_t n=multiExposureImages.size();
	for(size_t i=0;i<n;i++){
		Mat hsvImage;
		vector<Mat>planes;
		cvtColor(multiExposureImages[i],hsvImage,CV_RGB2HSV);
		cv::split(hsvImage,planes);
		outputImages[i]=planes[2];
	}
}
void getWeightMapImage_bright(vector<Mat>inputImages,vector<Mat>&outputImages){
	size_t n=inputImages.size();
	Size size=inputImages[0].size();
	int rows=size.height;
	int cols=size.width;
	Mat averageImage(rows,cols,CV_32FC1,Scalar::all(0));

	for(size_t i=0;i<n;i++){
		//imshow(to_string(long long(i)),inputImages[i]);
		inputImages[i].convertTo(inputImages[i],CV_32FC1,1.0/255);
		averageImage+=inputImages[i];
		//cout<<averageImage.type();
	}
	averageImage/=n;
	//imshow("averageImage",averageImage);

	for(size_t i=0;i<n;i++){
		Mat tmp(rows,cols,CV_32FC1,Scalar::all(1));
		outputImages[i]=tmp-(inputImages[i]-averageImage).mul((inputImages[i]-averageImage))/(255*255);
		imshow(to_string(long long(i)),255*outputImages[i]);
	}

	//归一化
	Mat sumImage(rows,cols,CV_32FC1,Scalar::all(0));
	for(size_t i=0;i<n;i++){
		sumImage+=outputImages[i];
	}
	for(size_t i=0;i<n;i++){
		outputImages[i]=outputImages[i].mul(sumImage);
	}
}


int main(){
	vector<Mat>multiexposure(num);
	string path="SubejctiveTest/lamp";
	loadMultiExposureImages(multiexposure,path,".jpg");
	Size size=multiexposure[0].size();
	int w=size.width;
	int h=size.height;
	//////////////
	vector<Mat>multiVchannelsImages(num);
	getVchannelOfHSV(multiexposure,multiVchannelsImages);
	vector<Mat>res(num);
	for(int i=0;i<num;i++){
		res[i].create(h,w,CV_32FC1);
	}
	getWeightMapImage_bright(multiVchannelsImages,res);
	for(size_t i=0;i<num;i++){
		//imshow(to_string(long long(i)),res[i]);
	}
	for(size_t i=0;i<num;i++){
		//imshow(to_string(long long(i)),multiVchannelsImages[i]);
	}

	//////////////////
	/*
	vector<Mat>multiContrast(num);
	getMultiContrastImage(multiexposure,multiContrast);


	vector<Mat>MapImage(num);
	for(int i=0;i<num;i++){
		MapImage[i].create(h,w,CV_32FC1);
	}

	getWeightMapImage_Contrast(multiContrast,MapImage);
	
	Mat exposureFusionImage=getExposureFusionImage(multiexposure,MapImage);
	imshow("exposureFusionImage",exposureFusionImage);
	*/
	
	waitKey(0);
}