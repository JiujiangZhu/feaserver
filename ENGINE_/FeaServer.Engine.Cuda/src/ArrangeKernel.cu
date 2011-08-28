__kernel void reduce(__global uint4* input, __global uint4* output, __local uint4* sdata)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);
    unsigned int gid = get_global_id(0);

    unsigned int localSize = get_local_size(0);
    unsigned int stride = gid * 2;
    sdata[tid] = input[stride] + input[stride + 1];

    barrier(CLK_LOCAL_MEM_FENCE);
    // do reduction in shared mem
    for (unsigned int s = localSize >> 1; s > 0; s >>= 1) 
    {
        if (tid < s) 
            sdata[tid] += sdata[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if (tid == 0)
		output[bid] = sdata[0];
}

