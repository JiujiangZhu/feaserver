#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
/*
//http://supercomputingblog.com/cuda/cuda-tutorial-1-getting-started/
//http://people.eku.edu/ritchisong/301notes2.htm

typedef struct {
	int state;
} Element0;

__global__ void In(char **lookup)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	Element0 *element0 = (Element0 *)lookup[0][x];
}

__global__ void Box(char **lookup)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	Element0 *element0 = (Element0 *)lookup[0][x];
	int s = element0->state;
}
*/