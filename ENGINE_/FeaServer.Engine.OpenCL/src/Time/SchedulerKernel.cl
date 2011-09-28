#include "Core.clh"

__kernel void Test() {
	printf("Test");
}

/*
__kernel void vectorAdd(__global const float *a, __global const float *b, __global float *c)
{
	int nIndex = get_global_id(0);
	c[nIndex] = a[nIndex] + b[nIndex];
}
*/