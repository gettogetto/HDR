#ifndef HDRALGORITHM_H
#define HDRALGORITHM_H
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<vector>
using namespace cv;
#define num 2
class hdrAlgorithm
{
public:
	hdrAlgorithm();
	~hdrAlgorithm();
	void getHdrImage(vector<Mat>&multiExposureImages_in,const string& path,const string& imageType,Mat&outputImage);
private:
	void loadMultiExposureImages(vector<Mat>&multiExposureImages,const string& path,const string& imageType);
	void doFilter(vector<Mat>&inputImages,vector<Mat>&outputImages);
	Mat getContrastImage(const Mat& srcImage,int midValue);
	void getMultiContrastImage(const vector<Mat>&multiExposureImages,vector<Mat>&multiContrastImages,int midValue);
	void getWeightMapImage_Contrast(const vector<Mat>& multiContrast,vector<Mat>& multiMapImage);

	void getVchannelOfHSV(const vector<Mat>& multiExposureImages,vector<Mat>& outputImages);
	void getWeightMapImage_bright(const vector<Mat>&inputImages,vector<Mat>&outputImages);
	
	void weightMapImage_use_Contrast_and_bright(const vector<Mat>&inputImages1,const vector<Mat>&inputImages2,vector<Mat>&outputImages);
	Mat getExposureFusionImage(const vector<Mat>&multiExposureImages,vector<Mat>&multiWeightMapImage);
private:
	vector<Mat>m_multiExposureImages;
	vector<Mat>m_multiContrast12;
	vector<Mat>m_multiContrast8;
};
#endif
