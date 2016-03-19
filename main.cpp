#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>
#include<iostream>
using namespace std;
using namespace cv;
#define num 7
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
		outputImages[i].convertTo(outputImages[i],CV_32FC1,1.0/255);
		//cout<<outputImages[i].type();
		//imshow(to_string(long long(i)),outputImages[i]);
	}
}
void getWeightMapImage_bright(vector<Mat>inputImages,vector<Mat>&outputImages){
	size_t n=inputImages.size();
	Size size=inputImages[0].size();
	int rows=size.height;
	int cols=size.width;
	Mat averageImage(rows,cols,CV_32FC1,Scalar::all(0));

	for(size_t i=0;i<n;i++){
		//cout<<inputImages[i].type()<<endl;
		//imshow(to_string(long long(i)),inputImages[i]);
		averageImage+=inputImages[i];
	}
	averageImage/=n;
	//imshow("averageImage",averageImage);
	//cout<<averageImage.at<float>(200,100);

	for(size_t i=0;i<n;i++){
		//outputImages[i]=1-(inputImages[i]-averageImage).mul(inputImages[i]-averageImage)*500000;

		//====================================================================//
		for(size_t r=0;r<rows;r++){
			for(size_t c=0;c<cols;c++){
				double diff=inputImages[i].at<float>(r,c)-averageImage.at<float>(r,c);
				double diff2=diff*diff;
				double sigma=n<=5?1.0/n:0.2;
				double mi=-diff2/(2*sigma*sigma);
				outputImages[i].at<float>(r,c)=std::pow(2.71828,mi);//高斯
			}
		}
			//cout<<outputImages[i].at<float>(100,100)<<endl;
		//imshow(to_string(long long(i)),outputImages[i]);
	}

	//归一化
	Mat sumImage(rows,cols,CV_32FC1,Scalar::all(0));
	for(size_t i=0;i<n;i++){
		sumImage+=outputImages[i];
	}
	//cout<<sumImage.at<float>(100,100);
	for(size_t i=0;i<n;i++){
		outputImages[i]=outputImages[i]/sumImage;
		//cout<<outputImages[i].at<float>(100,100);
		//imshow(to_string(long long(i)),outputImages[i]);
	}
}

void weightMapImage_use_Contrast_and_bright(const vector<Mat>&inputImages1,const vector<Mat>&inputImages2,vector<Mat>&outputImages){
	size_t n=inputImages1.size();
	size_t n2=inputImages2.size();

	if(n!=n2){
		std::cout<<"矩阵数量不同"<<endl;
		return;
	}
	int rows=inputImages1[0].size().height;
	int cols=inputImages1[0].size().width;

	for(size_t i=0;i<n;i++){
		outputImages[i]=inputImages1[i].mul(inputImages2[i]);
	}
	Mat sumImage(rows,cols,CV_32FC1,Scalar::all(0));
	for(size_t i=0;i<n;i++){
		sumImage+=outputImages[i];
	}
	for(size_t i=0;i<n;i++){
		outputImages[i]=outputImages[i]/sumImage;
	}
}

int main(){
	vector<Mat>multiexposure(num);
	string path="SubejctiveTest/14";
	loadMultiExposureImages(multiexposure,path,".jpg");
	Size size=multiexposure[0].size();
	int w=size.width;
	int h=size.height;
	//////////////
	vector<Mat>multiVchannelsImages(num);
	for(int i=0;i<num;i++){
		multiVchannelsImages[i].create(h,w,CV_32FC1);
	}
	getVchannelOfHSV(multiexposure,multiVchannelsImages);
	/////////////////////////////////////亮度比重图
	vector<Mat>MapImage_bright(num);
	for(int i=0;i<num;i++){
		MapImage_bright[i].create(h,w,CV_32FC1);
	}
	getWeightMapImage_bright(multiVchannelsImages,MapImage_bright);

	//////////////////细节比重图
	
	vector<Mat>multiContrast(num);
	getMultiContrastImage(multiexposure,multiContrast);


	vector<Mat>MapImage_Contrast(num);
	for(int i=0;i<num;i++){
		MapImage_Contrast[i].create(h,w,CV_32FC1);
	}

	getWeightMapImage_Contrast(multiContrast,MapImage_Contrast);
	/////////////////////////////////亮度细节比重图
	vector<Mat>MapImage_Contrast_and_bright(num);
	for(int i=0;i<num;i++){
		MapImage_Contrast[i].create(h,w,CV_32FC1);
	}
	weightMapImage_use_Contrast_and_bright(MapImage_bright,MapImage_Contrast,MapImage_Contrast_and_bright);

	////////////////////////
	Mat exposureFusionImage1=getExposureFusionImage(multiexposure,MapImage_bright);
	Mat exposureFusionImage2=getExposureFusionImage(multiexposure,MapImage_Contrast);
	Mat exposureFusionImage3=getExposureFusionImage(multiexposure,MapImage_Contrast_and_bright);
	imshow("MapImage_bright",exposureFusionImage1);
	imshow("MapImage_Contrast",exposureFusionImage2);
	imshow("MapImage_Contrast_and_bright",exposureFusionImage3);
	
	waitKey(0);
}