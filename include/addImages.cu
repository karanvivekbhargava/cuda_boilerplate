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
 *  @file    addImages.cu
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
#pragma once

// Global specifier indicates that the code is for the device side / GPU side
__global__

/// @brief addImages    takes in the x and y pointers and sums the data up (averaging each pixel from x and y)
///                     It changes the x memory to store the result.
/// @param n            The total memory size
/// @param x            The pointer to an array of image data
/// @param y            The pointer to another array of image data
void addImages(int n, unsigned char *x, unsigned char *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] = (x[i]/2) + (y[i]/2);
    }
}