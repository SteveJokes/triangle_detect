#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include "hough.h"
#include "math.h"
using namespace cv;
using namespace std;

#define PI 3.1415926
//全局变量声明
Mat srcImage, grayImage, dstImage, normImage, Scalednorm, resultImage;
int  thresh = 125;
int max_thresh = 255;
int tri_count=0;
int valid_tri[20][3];
double edge_len_max=0;
//函数声明
void HarrisConner_demo(InputArray _src, OutputArray _dst);
void get_triangle(InputArray _src, OutputArray _dst);
double line_len(int *start,int *end);
void find_vertex(int len,float threshold);
int edgePoint[30][2];
//主函数
int main(int argc, char** argv) {
    //读取图像
    //srcImage = imread("../data/triangle.png");
    srcImage= imread("../data/triangle_real.jpg");
    if (srcImage.empty()) {
        cout << "could not load srcimage...\n" << endl;
        return -1;
    }
    get_triangle(srcImage,dstImage);
    HarrisConner_demo(dstImage,dstImage);
    waitKey(0);
    return 0;
}

void find_vertex(int len,float threshold)
{
    double edge_len,len_to_mid;
    int edge_max[2];
    int mid_point[2];
    for(int i=0;i<len-1;i++)
    {
        for(int j=i+1;j<len;j++)
        {
            mid_point[0]=(edgePoint[i][0]+edgePoint[j][0])/2;
            mid_point[1]=(edgePoint[i][1]+edgePoint[j][1])/2;
            edge_len=line_len(edgePoint[i],edgePoint[j]);
            if(edge_len<edge_len_max)continue;
            if(edge_len>edge_len_max)
            {
                edge_len_max=edge_len;
                edge_max[0]=i;
                edge_max[1]=j;
            }
        }
    }
    for(int k=0;k<=len;k++)
    {
        if(k==edge_max[0]||k==edge_max[1])continue;
        len_to_mid=line_len(mid_point,edgePoint[k]);
        if(abs(len_to_mid-edge_len/2)/edge_len<threshold)
        {
            valid_tri[tri_count][0]=edge_max[0];
            valid_tri[tri_count][1]=edge_max[1];
            valid_tri[tri_count][2]=k;
            //tri_count++;
        }
        else
            cout<<"no triangle! when k="<<k<<endl;
    }
    //return valid_tri;
}
double line_len(int *start,int *end)
{
    double len=sqrt(pow(start[0]-end[0],2)+pow(start[1]-end[1],2));
    return len;
}

void HarrisConner_demo(InputArray _src, OutputArray _dst) {
    //置零
    Mat input_Image=_src.getMat();
    dstImage = Mat::zeros(input_Image.size(), CV_32FC1);
    //参数赋值
    int blocksize = 2;
    int ksize = 3;
    double k = 0.04;
    int edgePointLast[2];
    int tri_vertex[3][2];
    int pointCount=0;
    cornerHarris(input_Image, dstImage, blocksize, ksize, k, BORDER_DEFAULT);
    normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    //将归一化后的图线性变换成8位无符号整型
    convertScaleAbs(normImage, Scalednorm);
    resultImage = srcImage.clone();
    for (int i = 0; i < resultImage.rows; i++) {
         //uchar* pixel_i = Scalednorm.ptr(i);
        for (int j = 0; j < resultImage.cols; j++) {
                if ((int)normImage.at<float>(i, j)>100&&(abs(edgePointLast[0]-j)+abs(edgePointLast[1]-i))>5) {
                    cout<<j<<i<<endl;
                    edgePoint[pointCount][0]=j;
                    edgePoint[pointCount][1]=i;
                    //circle(resultImage, Point(j, i), 4, Scalar(0, 0, 255), 2, 8, 0);
                    circle(Scalednorm, Point(j, i), 5, Scalar(1, 10, 255), 2, 8, 0);
                    edgePointLast[0]=j;
                    edgePointLast[1]=i;
                    pointCount+=1;

            }
                //pixel_i++;
        }
    }
    //int *vertex;
    find_vertex(pointCount,0.2);
    for(int i=0;i<=tri_count;i++){
        //valid_tri
        Point2i p_tri_vertex[3];
        int n_tri_vertex[3][2];
        for(int j=0;j<3;j++)
        {
            n_tri_vertex[j][0]=edgePoint[valid_tri[i][j]][0];
            n_tri_vertex[j][1]=edgePoint[valid_tri[i][j]][1];
            p_tri_vertex[j]=Point(edgePoint[valid_tri[i][j]][0],edgePoint[valid_tri[i][j]][1]);
        }
        circle(resultImage, Point(edgePoint[valid_tri[i][0]][0],edgePoint[valid_tri[i][0]][1]), 4, Scalar(0, 0, 255), 2, 8, 0);
        circle(resultImage, Point(edgePoint[valid_tri[i][1]][0],edgePoint[valid_tri[i][1]][1]), 4, Scalar(0, 0, 255), 2, 8, 0);
        circle(resultImage, Point(edgePoint[valid_tri[i][2]][0],edgePoint[valid_tri[i][2]][1]), 4, Scalar(0, 0, 255), 2, 8, 0);
        float line_end_point[2][2],end_point[2];
        line_end_point[0][0]=-(n_tri_vertex[0][0]-n_tri_vertex[2][0])*edge_len_max*0.5/line_len(n_tri_vertex[0],n_tri_vertex[2]);
        line_end_point[0][1]=-(n_tri_vertex[0][1]-n_tri_vertex[2][1])*edge_len_max*0.5/line_len(n_tri_vertex[0],n_tri_vertex[2]);
        line_end_point[1][0]=-(n_tri_vertex[1][0]-n_tri_vertex[2][0])*edge_len_max*0.5/line_len(n_tri_vertex[1],n_tri_vertex[2]);
        line_end_point[1][1]=-(n_tri_vertex[1][1]-n_tri_vertex[2][1])*edge_len_max*0.5/line_len(n_tri_vertex[1],n_tri_vertex[2]);
        end_point[0]=n_tri_vertex[2][0]+line_end_point[0][0]+line_end_point[1][0];
        end_point[1]=n_tri_vertex[2][1]+line_end_point[0][1]+line_end_point[1][1];
        Point p_end_point(end_point[0],end_point[1]);
        line(resultImage, p_tri_vertex[2],p_end_point, Scalar(255, 0, 255),5);
        line(Scalednorm, p_tri_vertex[2],p_end_point, Scalar(255, 0, 255),5);
    }

    _dst.getMatRef() = resultImage;
    //显示角点检测的结果
    imshow("HarrisCornerDetection", resultImage);
    imshow("Detection", Scalednorm);
    waitKey(0);
}


void get_triangle(InputArray _src, OutputArray _dst)
{
    using namespace cv;
    Mat input_image = _src.getMat();
    Mat blur_img;
    Mat edge_img;
    //Mat phology_out;
    vector<Vec4i>lines;
    vector<vector<Point> > contours;

    Mat output_image;
    Mat imgHSV;

    const Scalar hsvRedLo1( 0,  40,  40);
    const Scalar hsvRedHi1(10, 255, 255);
    const Scalar hsvRedLo2( 156,  40,  40);
    const Scalar hsvRedHi2(180, 255, 255);

    blur(input_image,blur_img,Size(3,3));//均值滤波
    cvtColor(blur_img, imgHSV, COLOR_BGR2HSV);//转为HSV
    Mat imgThresholded,imgThresholded2,redImg;
    inRange(imgHSV,hsvRedLo1,hsvRedHi1, imgThresholded); //滤出红色部分
    inRange(imgHSV,hsvRedLo2,hsvRedHi2, imgThresholded2);
    addWeighted(imgThresholded,1,imgThresholded2,1, 0.0,redImg);
    Mat element = getStructuringElement(MORPH_RECT, Size(5,5));
    morphologyEx(redImg,redImg, MORPH_CLOSE, element);//闭操作 (连接一些连通域)
    morphologyEx(redImg,redImg, MORPH_OPEN, element);//开操作 (去除一些噪点)
    output_image = input_image.clone();
    _dst.getMatRef() =redImg;
/****************以下测试观察用**********************/
    Canny(redImg,edge_img, 90, 150, 3);//提取边缘

    //imshow("image",redImg);
    //imshow("image2",edge_img);
    //line detect
    HoughLinesP(edge_img, lines, 1, (PI / 180),10,180,150);
    for( size_t i = 0; i < lines.size(); i++ )
        {
            line( output_image, Point(lines[i][0], lines[i][1]),
                Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
        }
    imshow("output_image",output_image);
    return ;
}
