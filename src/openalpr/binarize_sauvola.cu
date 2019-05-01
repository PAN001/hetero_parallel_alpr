/**************************************************************
 * Binarization with several methods
 * (0) Niblacks method
 * (1) Sauvola & Co.
 *     ICDAR 1997, pp 147-152
 * (2) by myself - Christian Wolf
 *     Research notebook 19.4.2001, page 129
 * (3) by myself - Christian Wolf
 *     20.4.2007
 *
 * See also:
 * Research notebook 24.4.2001, page 132 (Calculation of s)
 **************************************************************/

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <iostream>
// #include <cv>
// #include <highgui>
#include <opencv2/opencv.hpp>
#include "timing.h"

using namespace std;
using namespace cv;

enum NiblackVersion
{
	NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};

#define BINARIZEWOLF_VERSION	"2.4 (August 1st, 2014)"
#define BINARIZEWOLF_DEFAULTDR	128

#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;


static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
    if(err!=cudaSuccess)
    {
        fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
        std::cin.get();
        exit(EXIT_FAILURE);
    }
}

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)



// *************************************************************
double calcLocalStats (Mat &im, Mat &im_sum, Mat &im_sum_sq, Mat &map_m, Mat &map_s, int winx, int winy) {    
    timespec startTime;
    getTimeMonotonic(&startTime);

    // Mat im_sum, im_sum_sq;
    // cv::integral(im,im_sum,im_sum_sq,CV_64F); // TODO: no need to calculated this everytime

    // timespec endTime;
    // getTimeMonotonic(&endTime);
    // cout << "cv::integral Time: " << diffclock(startTime, endTime) << "ms." << endl;

    // double m,s,max_s,sum,sum_sq;  
    int wxh   = winx/2;
    int wyh   = winy/2;
    int x_firstth= wxh;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    double winarea = winx*winy;

    double max_s = 0;

    // cout << "  --outerloop size: " << y_lastth - y_firstth << endl;
    // cout << "  --innerloop size: " << im.cols-winx - 1 << endl;
    for(int j = y_firstth ; j<=y_lastth; j++) { 
        double m,s,sum,sum_sq;  
        sum = sum_sq = 0;

        // sum of the window
        sum = im_sum.at<double>(j-wyh+winy,winx) - im_sum.at<double>(j-wyh,winx) - im_sum.at<double>(j-wyh+winy,0) + im_sum.at<double>(j-wyh,0);
        sum_sq = im_sum_sq.at<double>(j-wyh+winy,winx) - im_sum_sq.at<double>(j-wyh,winx) - im_sum_sq.at<double>(j-wyh+winy,0) + im_sum_sq.at<double>(j-wyh,0);

        m  = sum / winarea;
        s  = sqrt ((sum_sq - m*sum)/winarea);
        if (s > max_s) max_s = s;

        map_m.fset(x_firstth, j, m);
        map_s.fset(x_firstth, j, s);

        // Shift the window, add and remove   new/old values to the histogram
        for(int i=1 ; i <= im.cols-winx; i++) {
            // Remove the left old column and add the right new column
            sum -= im_sum.at<double>(j-wyh+winy,i) - im_sum.at<double>(j-wyh,i) - im_sum.at<double>(j-wyh+winy,i-1) + im_sum.at<double>(j-wyh,i-1);
            sum += im_sum.at<double>(j-wyh+winy,i+winx) - im_sum.at<double>(j-wyh,i+winx) - im_sum.at<double>(j-wyh+winy,i+winx-1) + im_sum.at<double>(j-wyh,i+winx-1);

            sum_sq -= im_sum_sq.at<double>(j-wyh+winy,i) - im_sum_sq.at<double>(j-wyh,i) - im_sum_sq.at<double>(j-wyh+winy,i-1) + im_sum_sq.at<double>(j-wyh,i-1);
            sum_sq += im_sum_sq.at<double>(j-wyh+winy,i+winx) - im_sum_sq.at<double>(j-wyh,i+winx) - im_sum_sq.at<double>(j-wyh+winy,i+winx-1) + im_sum_sq.at<double>(j-wyh,i+winx-1);

            m  = sum / winarea;
            s  = sqrt ((sum_sq - m*sum)/winarea);
            if (s > max_s) max_s = s;

            map_m.fset(i+wxh, j, m);
            map_s.fset(i+wxh, j, s);
        }
    }


    timespec endTime;
    getTimeMonotonic(&endTime);
    cout << "  --calcLocalStats Time: " << diffclock(startTime, endTime) << "ms." << endl;

    return max_s;
}

/**********************************************************
* The binarization routine
**********************************************************/
// https://pdfs.semanticscholar.org/0695/bb92a3301be9343433334692bd54c31a8233.pdf:
// Niblack’s algorithm calculates a threshold sur- face by gliding a rectangular window across the image. 
// The threshold T for the center pixel of the window is computed using the mean m and the variance s of the gray values in the window:
// T = m + k · s, where k is a constant set to −0.2.

__device__ __inline__ void set_color(unsigned char* input, unsigned char* output, int row_idx, int col_idx, double th, int width_step) {
    int idx = row_idx * width_step + col_idx;
    if(input[idx] >= th) {
        output[idx] = 255; // set to white
    }
    else {
        output[idx] = 0;
    }
}

__global__ void NiblackSauvolaWolfJolionCuda(unsigned char* input, unsigned char* output, int winx, int winy, double k, int img_width, int img_height, int width_step, int rows_per_thread) {
    double m, s, sum, sum_sq, foo;
    double th=0;
    // double min_I, max_I;
    int wxh = winx/2;
    int wyh = winy/2;
    int x_firstth= wxh;
    int x_lastth = img_width-wxh-1;
    int y_lastth = img_height-wyh-1;
    int y_firstth= wyh;
    double winarea = winx*winy;

    // Mat thsurf (im.rows, im.cols, CV_32F);
            
    // Create the threshold surface, including border processing
    // ----------------------------------------------------

    int row_start_idx = blockIdx.x * blockDim.x + threadIdx.x; // row index
    // printf("raw row_start_idx: %d\n", row_start_idx);
    row_start_idx *= rows_per_thread;
    row_start_idx += y_firstth;
    int row_end_idx = row_start_idx + rows_per_thread;
    row_end_idx = row_end_idx - 1 > y_lastth ? y_lastth + 1 : row_end_idx;

    if(row_start_idx > y_lastth)
        return;

    for(int row_idx = row_start_idx;row_idx < row_end_idx;row_idx++) {
        // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW

        // Calculate the initial window at the beginning of the line
        sum = sum_sq = 0;
        for(int wy=0 ; wy<winy; wy++) {
            for(int wx=0 ; wx<winx; wx++) {
                // foo = im.uget(wx,j-wyh+wy);
                foo = input[(row_idx-wyh+wy) * width_step + wx];
                sum += foo;
                sum_sq += foo*foo;
            }
        }

        m = sum / winarea;
        s = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);

        th = m * (1 + k*(s/BINARIZEWOLF_DEFAULTDR-1));
        set_color(input, output, row_idx, 0+wxh, th, width_step);

        // LEFT BORDER
        for (int i=0; i<=x_firstth; ++i)
            set_color(input, output, row_idx, i, th, width_step);

        // LEFT-UPPER CORNER
        if (row_idx==y_firstth)
            for (int u=0; u<y_firstth; ++u)
                for (int i=0; i<=x_firstth; ++i)
                    set_color(input, output, u, i, th, width_step);

        // LEFT-LOWER CORNER
        if (row_idx==y_lastth)
            for (int u=y_lastth+1; u<img_height; ++u)
                for (int i=0; i<=x_firstth; ++i)
                    set_color(input, output, u, i, th, width_step);

        for (int i=1 ; i <= img_width-winx; i++) {
            // Remove the left old column and add the right new column
            for(int wy=0; wy<winy; ++wy) {
                // foo = im.uget(i-1,j-wyh+wy);
                foo =  input[(row_idx-wyh+wy) * width_step + i - 1];
                sum -= foo;
                sum_sq -= foo*foo;
                // foo = im.uget(i+winx-1,j-wyh+wy);
                foo =  input[(row_idx-wyh+wy) * width_step + i + winx - 1];
                sum += foo;
                sum_sq += foo*foo;
            }

            m = sum / winarea;
            s = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
            th = m * (1 + k*(s/BINARIZEWOLF_DEFAULTDR-1));
            set_color(input, output, row_idx, i+wxh, th, width_step);

            // UPPER BORDER
            if (row_idx==y_firstth)
                for (int u=0; u<y_firstth; ++u)
                    set_color(input, output, u, i+wxh, th, width_step);

            // LOWER BORDER
            if (row_idx==y_lastth)
                for (int u=y_lastth+1; u<img_height; ++u)
                    set_color(input, output, u, i+wxh, th, width_step);
        }

        // RIGHT BORDER
        for (int i=x_lastth; i<img_width; ++i)
            set_color(input, output, row_idx, i, th, width_step);

        // RIGHT-UPPER CORNER
        if (row_idx==y_firstth)
            for (int u=0; u<y_firstth; ++u)
                for (int i=x_lastth; i<img_width; ++i)
                    set_color(input, output, u, i, th, width_step);

        // RIGHT-LOWER CORNER
        if (row_idx==y_lastth)
            for (int u=y_lastth+1; u<img_height; ++u)
                for (int i=x_lastth; i<img_width; ++i)
                    set_color(input, output, u, i, th, width_step);
    }
}

void NiblackSauvolaWolfJolionWrapper(Mat input, Mat output, int winx, int winy, double k) {
    timespec contextStartTime;
    getTimeMonotonic(&contextStartTime);

    cudaFree(0); // manually trigger creation of the context

    timespec endTime;
    getTimeMonotonic(&endTime);
    cout << "  --context creation Time: " << diffclock(contextStartTime, endTime) << "ms." << endl;

    timespec startTime;
    getTimeMonotonic(&startTime);

    // Mat im_sum, im_sum_sq;
    // integral(input, im_sum, im_sum_sq, CV_64F);

    // timespec integralEndTime;
    // getTimeMonotonic(&integralEndTime);
    // cout << "  --cv::integral Time: " << diffclock(startTime, integralEndTime) << "ms." << endl;


    // timespec minMaxLocStartTime;
    // getTimeMonotonic(&minMaxLocStartTime);
    // double min_I, max_I;
    // minMaxLoc(input, &min_I, &max_I);
    // timespec minMaxLocEndTime;
    // getTimeMonotonic(&minMaxLocEndTime);
    // cout << "  --cv::minMaxLoc Time: " << diffclock(minMaxLocStartTime, minMaxLocEndTime) << "ms." << endl;


    // // Create local statistics and store them in a double matrices
    // Mat map_m = Mat::zeros(input.rows, input.cols, CV_32F); // mean of the gray values in the window
    // Mat map_s = Mat::zeros(input.rows, input.cols, CV_32F); // variance of the gray values in the window
    // double max_s = calcLocalStats(input, im_sum, im_sum_sq, map_m, map_s, winx, winy);

    // cout << "input.rows: " << input.rows << endl;
    // cout << "input.step: " << input.step << endl;

    //Calculate total number of bytes of input and output image
    const int inputBytes = input.step * input.rows;
    const int outputBytes = output.step * output.rows;
    // const int sumBytes = im_sum.cols * im_sum.rows;
    // const int sumSqBytes = im_sum_sq.cols * im_sum_sq.rows;
    // const int mapBytes = map_m.step * map_m.rows;

    unsigned char *d_input, *d_output;
    // float *d_sum, *d_sum_sq;
    // float *d_map_m, *d_map_s;

    timespec cudaMallocStartTime;
    getTimeMonotonic(&cudaMallocStartTime);
    //Allocate device memory
    SAFE_CALL(cudaMalloc<unsigned char>(&d_input,inputBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc<unsigned char>(&d_output,outputBytes),"CUDA Malloc Failed");
    // SAFE_CALL(cudaMalloc<unsigned char>(&d_sum,sumBytes),"CUDA Malloc Failed");
    // SAFE_CALL(cudaMalloc<unsigned char>(&d_sum_sq,sumSqBytes),"CUDA Malloc Failed");
    // SAFE_CALL(cudaMalloc<float>(&d_map_m,mapBytes),"CUDA Malloc Failed");
    // SAFE_CALL(cudaMalloc<float>(&d_map_s,mapBytes),"CUDA Malloc Failed");

    getTimeMonotonic(&endTime);
    cout << "  --cudaMalloc Time: " << diffclock(cudaMallocStartTime, endTime) << "ms." << endl;

    timespec cudaMemcpyStartTime;
    getTimeMonotonic(&cudaMemcpyStartTime);
    //Copy data from OpenCV input image to device memory
    SAFE_CALL(cudaMemcpy(d_input,input.ptr(),inputBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    // SAFE_CALL(cudaMemcpy(d_sum,im_sum.ptr(),sumBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    // SAFE_CALL(cudaMemcpy(d_sum_sq,im_sum_sq.ptr(),sumSqBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    // SAFE_CALL(cudaMemcpy(d_map_m,map_m.ptr(),mapBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    // SAFE_CALL(cudaMemcpy(d_map_s,map_s.ptr(),mapBytes,cudaMemcpyHostToDevice),"CUDA Memcpy Host To Device Failed");
    getTimeMonotonic(&endTime);
    cout << "  --cudaMemcpy Time: " << diffclock(cudaMemcpyStartTime, endTime) << "ms." << endl;

    //1-d allocation
    int wyh = winy/2;
    int y_lastth = input.rows-wyh-1;
    // cout << "y_lastth: " << y_lastth << endl;
    int y_firstth= wyh;
    // cout << "y_firstth: " << y_firstth << endl;
    int total_cnt = y_lastth - y_firstth + 1;
    int rows_per_thread = 8;
    // cout << "total_cnt: " << total_cnt << endl;
    const dim3 block(32 / rows_per_thread, 1, 1);
    // cout << "block.x: " << block.x << endl;
    int gridX = (total_cnt + block.x * rows_per_thread - 1) / (block.x * rows_per_thread);
    const dim3 grid(gridX, 1, 1);
    // cout << "grid.x: " << grid.x << endl;

    getTimeMonotonic(&endTime);
    cout << "  --cuda data preparing kernel Time: " << diffclock(cudaMallocStartTime, endTime) << "ms." << endl;

    timespec cudaKernelStartTime;
    getTimeMonotonic(&cudaKernelStartTime);

    //Launch the binarization kernel
    NiblackSauvolaWolfJolionCuda<<<grid,block>>>(d_input, d_output, winx, winy, k, input.cols, input.rows, input.step, rows_per_thread);

    //Synchronize to check for any kernel launch errors
    SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    //Copy back data from destination device meory to OpenCV output image
    SAFE_CALL(cudaMemcpy(output.ptr(),d_output,outputBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

    getTimeMonotonic(&endTime);
    cout << "  --cuda kernel running Time: " << diffclock(cudaKernelStartTime, endTime) << "ms." << endl;


    //Free the device memory
    SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
    // // SAFE_CALL(cudaFree(d_sum),"CUDA Free Failed");
    // // SAFE_CALL(cudaFree(d_sum_sq),"CUDA Free Failed");
    // SAFE_CALL(cudaFree(d_map_m),"CUDA Free Failed");
    // SAFE_CALL(cudaFree(d_map_s),"CUDA Free Failed");

    getTimeMonotonic(&endTime);
    cout << "=========== Total Time (excluding CUDA context creation): " << diffclock(startTime, endTime) << "ms." << endl;
}
