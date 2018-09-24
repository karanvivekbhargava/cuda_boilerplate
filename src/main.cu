/**
 *  MIT License
 *
 *  Copyright (c) 2017 Karan Vivek Bhargava
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a
 *  copy of this software and associated documentation files (the "Software"),
 *  to deal in the Software without restriction, including without
 *  limitation the rights to use, copy, modify, merge, publish, distribute,
 *  sublicense, and/or sell copies of the Software, and to permit persons to
 *  whom the Software is furnished to do so, subject to the following
 *  conditions:
 *
 *  The above copyright notice and this permission notice shall be included
 *  in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 *  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 *  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 *  DEALINGS IN THE SOFTWARE.
 *
 *  @file    main.cu
 *  @author  Karan Vivek Bhargava
 *  @copyright MIT License
 *
 *  @brief CUDA Programming Boilerplate Code
 *
 *  @section DESCRIPTION
 *
 *  This program will open two images 'start' and 'end' inside the images folder and
 *  add them up on the GPU.
 *
 */


#include <iostream>                     // For debugging and printing
#include <opencv2/imgcodecs.hpp>        // For the various codecs
#include <opencv2/highgui/highgui.hpp>  // For showing the images
#include "addImages.cu"                 // Conatins the GPU side code to add the images

/// @brief The main function
int main(void)
{
    // Import the image into a cv::Mat container
    cv::Mat img1 = cv::imread("../images/start.jpg");
    cv::Mat img2 = cv::imread("../images/end.jpg");

    // Declare the required pointers to data in the host and device memory
    unsigned char *x, *y, *d_x, *d_y, *output;

    // Get the sizes of the images you're operating on
    int32_t size1 = 3 * img1.rows * img1.cols * sizeof(unsigned char);
    int32_t size2 = 3 * img2.rows * img2.cols * sizeof(unsigned char);

    // Create the required space for the resulting output after the addition
    output = (unsigned char *)malloc(size1);

    // Assign the pointers to the image data
    x = (unsigned char *)img1.data;
    y = (unsigned char *)img2.data;

    // Allocate the required memory on the gpu for the two images
    cudaMalloc(&d_x, size1);
    cudaMalloc(&d_y, size2);

    // Copy the elements of the images onto the cuda memory
    cudaMemcpy(d_x, x, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size2, cudaMemcpyHostToDevice);

    // Perform simple addition on the two matrices
    addImages<<<(size1+1023)/1024, 1024>>>(size1, d_x, d_y);

    // Copy the result back to the cpu memory
    cudaMemcpy(output, d_x, size1, cudaMemcpyDeviceToHost);

    // Store the result into an cv::Mat container for viewing
    cv::Mat result(img1.rows, img1.cols, img1.type(), output);

    // View the result
    cv::imshow("Result", result);
    cv::waitKey(500);

    // Free the memory from the GPU / device
    cudaFree(d_x);
    cudaFree(d_y);
    
    // Free the memory from the CPU / host
    free(output);
}