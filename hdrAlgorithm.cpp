#include "hdrAlgorithm.h"


hdrAlgorithm::hdrAlgorithm()
{

}


hdrAlgorithm::~hdrAlgorithm()
{

}
void hdrAlgorithm::getHdrImage(vector<Mat>&multiExposureImages_in,const string& path,const string& imageType,Mat& outputImage){
	loadMultiExposureImages(multiExposureImages_in,path,imageType);
	vector<Mat> multiExposureImages(multiExposureImages_in.size());
	doFilter(multiExposureImages_in,multiExposureImages);

	Size size=multiExposureImages[0].size();
	int w=size.width;
	int h=size.height;

	vector<Mat>multiVchannelsImages(num);
	for(int i=0;i<num;i++){
		multiVchannelsImages[i].create(h,w,CV_32FC1);
	}
	getVchannelOfHSV(multiExposureImages,multiVchannelsImages);
	/////////////////////////////////////亮度比重图
	vector<Mat>MapImage_bright(num);
	for(int i=0;i<num;i++){
		MapImage_bright[i].create(h,w,CV_32FC1);
	}
	getWeightMapImage_bright(multiVchannelsImages,MapImage_bright);
	//////////////////细节比重图
	vector<Mat>multiContrast(num);
	getMultiContrastImage(multiExposureImages,multiContrast);
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
	
	outputImage=getExposureFusionImage(multiExposureImages,MapImage_Contrast_and_bright);
}


Mat hdrAlgorithm::getContrastImage(const Mat& srcImage){
	Mat dstImage;
	/*
	Mat kenel(5,5,CV_32F,Scalar::all(0));
	kenel.at<float>(0,0)=-2;kenel.at<float>(0,1)=-4;kenel.at<float>(0,2)=-4;kenel.at<float>(0,3)=-4;kenel.at<float>(0,4)=-2;
	kenel.at<float>(1,0)=-4;kenel.at<float>(1,1)=0;kenel.at<float>(1,2)=8;kenel.at<float>(1,3)=0;kenel.at<float>(1,4)=-4;
	kenel.at<float>(2,0)=-4;kenel.at<float>(2,1)=8;kenel.at<float>(2,2)=100;kenel.at<float>(2,3)=8;kenel.at<float>(2,4)=-4;
	kenel.at<float>(3,0)=-4;kenel.at<float>(3,1)=0;kenel.at<float>(3,2)=8;kenel.at<float>(3,3)=0;kenel.at<float>(3,4)=-4;
	kenel.at<float>(4,0)=-2;kenel.at<float>(4,1)=-4;kenel.at<float>(4,2)=-4;kenel.at<float>(4,3)=-4;kenel.at<float>(4,4)=-2;
	*/
	Mat kenel(3,3,CV_32F,Scalar::all(0));
	kenel.at<float>(0,1)=-1;
	kenel.at<float>(1,0)=-1;
	kenel.at<float>(1,1)= 10;
	kenel.at<float>(1,2)=-1;
	kenel.at<float>(2,1)=-1;
	filter2D(srcImage,dstImage,srcImage.depth(),kenel);
	return dstImage;
}

void hdrAlgorithm::loadMultiExposureImages(vector<Mat>&multiExposureImages,const string& path,const string& imageType){
	size_t n=multiExposureImages.size();
	for(std::size_t i=0;i<n;i++){
		multiExposureImages[i]=imread(path+"/"+std::to_string(long long(i+1))+imageType);
	}
}

void hdrAlgorithm::doFilter(vector<Mat>&inputImages,vector<Mat>&outputImages){
	int n=inputImages.size();
	if(n==0) return ;
	int rows=inputImages[0].rows;
	int cols=inputImages[0].cols;
	//outputImages.reserve(n);
	//std::for_each(outputImages.begin(),outputImages.end(),[=](Mat& outputImage){outputImage.create(rows,cols,inputImages[0].type());});
	for(size_t i=0;i<n;i++){
		//两个高斯公式的σ可以相同，而且如果σ小于10，则滤波效果不明显，如果大于150，则会有强烈的卡通效果。当实时处理时，内核尺寸d推荐为5；
		//如果在非实时处理情况下，而且有较强的噪声时，d为9效果会较好。
		cv::bilateralFilter(inputImages[i],outputImages[i],5,10,10);
	}
	
}
void hdrAlgorithm::getMultiContrastImage(const vector<Mat>&multiExposureImages,vector<Mat>&multiContrastImages){
	for(int i=0;i<multiExposureImages.size();i++){
		Mat grayImage;
		cvtColor(multiExposureImages[i],grayImage,CV_BGR2GRAY);
		multiContrastImages[i]=getContrastImage(grayImage);
	}
}
void hdrAlgorithm::getWeightMapImage_Contrast(const vector<Mat>& multiContrast,vector<Mat>& multiMapImage){
	int n=multiContrast.size();
	//Size size=multiContrast[0].size();
	int w=multiContrast[0].cols;
	int h=multiContrast[0].rows;
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

Mat hdrAlgorithm::getExposureFusionImage(const vector<Mat>&multiExposureImages,vector<Mat>&multiWeightMapImage){

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


/////////////////////////////////////////////////////
void hdrAlgorithm::getVchannelOfHSV(const vector<Mat>& multiExposureImages,vector<Mat>& outputImages){
	size_t n=multiExposureImages.size();
	vector<Mat>planes;
	Mat hsvImage;
	for(size_t i=0;i<n;i++){
		cvtColor(multiExposureImages[i],hsvImage,CV_RGB2HSV);
		cv::split(hsvImage,planes);
		outputImages[i]=0.2*planes[0]+0.3*planes[1]+0.5*planes[2];
		outputImages[i].convertTo(outputImages[i],CV_32FC1,1.0/255);
		//cout<<outputImages[i].type();
		//imshow(std::to_string(long long(i)),outputImages[i]);
	}
}
void hdrAlgorithm::getWeightMapImage_bright(const vector<Mat>&inputImages,vector<Mat>&outputImages){
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
				double sigma=0.5;
				double mi=-diff2/(2*sigma*sigma);
				outputImages[i].at<float>(r,c)=std::pow(2.71828,mi);//高斯
			}
		}
			//cout<<outputImages[i].at<float>(100,100)<<endl;
		//imshow(std::to_string(long long(i)),outputImages[i]);
	}

	//归一化
	Mat sumImage(rows,cols,CV_32FC1,Scalar::all(0));
	for(size_t i=0;i<n;i++){
		sumImage+=outputImages[i];
	}
	//cout<<sumImage.at<float>(100,100);
	for(size_t i=0;i<n;i++){
		outputImages[i]=outputImages[i]/sumImage;
		//std::cout<<outputImages[i].at<float>(130,160);
		//imshow(std::to_string(long long(i)),outputImages[i]);
	}
}

void hdrAlgorithm::weightMapImage_use_Contrast_and_bright(const vector<Mat>&inputImages1,const vector<Mat>&inputImages2,vector<Mat>&outputImages){
	size_t n=inputImages1.size();
	size_t n2=inputImages2.size();
	const int bi = 0.8;
	assert(n==n2);

	int rows=inputImages1[0].rows;
	int cols=inputImages1[0].cols;

	for(size_t i=0;i<n;i++){
		outputImages[i]=bi*inputImages1[i]+(inputImages2[i]);
	}
	Mat sumImage(rows,cols,CV_32FC1,Scalar::all(0));
	for(size_t i=0;i<n;i++){
		sumImage+=outputImages[i];
	}
	for(size_t i=0;i<n;i++){
		outputImages[i]=outputImages[i]/sumImage;
	}
}