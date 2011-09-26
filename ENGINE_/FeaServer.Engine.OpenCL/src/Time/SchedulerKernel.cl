#include <cuda.h>;
#include "..\Core.h";
#include "..\..\..\FeaServer.Engine.Cpu\src\Time\Scheduler\SliceCollection.hpp"
using namespace Time::Scheduler;

__global__ void Schedule()
{
	Element e;
	e.ScheduleStyle = Time::ElementScheduleStyle::Multiple;

	SliceCollection s;
	s.Schedule(&e, 10);
	s.MoveNextSlice();
}

int main()
{
	//fallocHeapInitialize(nullptr, 0);

	cudaPrintfInit();
	Schedule<<<1, 1>>>();
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();
	printf("\ndone.\n");
	scanf("%c");
    return 0;
}

#include "..\..\..\FeaServer.Engine.Cpu\src\Time\Scheduler\SliceCollection.hpp"