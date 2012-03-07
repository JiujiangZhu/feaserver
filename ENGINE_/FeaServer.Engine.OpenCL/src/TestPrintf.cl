#include "Core.clh"
#include "System\clPrintf.cl"

__constant int testMessage[] = {'t','e','s','t'};

__kernel void Test(__global int* printHeap) { 
	clPrintf(nullptr, testMessage);
}
