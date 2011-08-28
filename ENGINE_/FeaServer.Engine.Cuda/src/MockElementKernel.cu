//http://people.eku.edu/ritchisong/301notes2.htm

typedef struct {
	int state;
} Element0;

__device__ void In(char **lookup)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	Element0 *element0 = (Element0 *)lookup[0][x];
}

__device__ void Box(char **lookup)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	Element0 *element0 = (Element0 *)lookup[0][x];
	int s = element0->state;
}
