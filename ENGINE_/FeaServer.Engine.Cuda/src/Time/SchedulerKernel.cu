#include <cuda.h>;
#include "..\Core.h";
#include "..\..\..\FeaServer.Engine.Cpu\src\Time\Scheduler\SliceCollection.hpp"
using namespace Time::Scheduler;

__global__ void Main()
{
	Element e;

	SliceCollection* s;
	s->Schedule(&e, 10);
	s->MoveNextSlice();

	//Element e;
	//ElementList* l;
	//l->AddFirst(&e);
	//ElementList::Enumerator en;
	//l->GetEnumerator(&en);
	//HibernateCollection h;
	//h.DeHibernate(nullptr);
}

#include "..\..\..\FeaServer.Engine.Cpu\src\Time\Scheduler\SliceCollection.hpp"