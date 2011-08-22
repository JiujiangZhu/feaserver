#include "Engine.h"
using namespace System;
namespace TimeServices { namespace Engine {
	void Engine::HibernateValue(LinkKind* value, unsigned long time)
	{
		//Console::WriteLine(L"Timeline: Hibernate " + time.ToString());
		value->HibernateTime = time;
		LinkKind::AddFirst(_hibernateSegments[0].Chain, value);
	}

    void Engine::HibernateValue(ListKind* value, unsigned long time)
    {
		//Console::WriteLine(L"Timeline: Hibernate " + time.ToString());
		//_hibernateSegments[0].List.AddFirst(new HibernateState { Time = time, Object = value });
    }
   
    void Engine::DehibernateAnyValues()
    {
        //var list = _hibernateSegments[0].List;
        //if (list.Count > 0)
        //{
        //    for (var node = list.First; node != null; node = node.Next)
        //    {
        //        var hibernateState = node.Value;
        //        if (hibernateState.Time < EngineSettings::MaxTimeslicesTime)
        //            throw new InvalidOperationException(); //+ paranoia
        //        unsigned long newTime = (hibernateState.Time -= MaxTimeslicesTime);
        //        if (newTime < EngineSettings::MaxTimeslicesTime)
        //        {
        //            //Console::WriteLine(L"Timeline: Dehibernate {" + newTime.ToString() + "}");
        //            // remove node
        //            list.Remove(node);
        //            // add to timeline
        //            AddValue(hibernateState.Object, newTime);
        //        }
        //    }
        //}
        LinkKind* chain = _hibernateSegments[0].Chain;
        if (chain != nullptr)
        {
            LinkKind* lastItem = chain;
            for (LinkKind* nextItem = lastItem; nextItem != nullptr; nextItem = nextItem->NextLink)
            {
                if (nextItem->HibernateTime < EngineSettings::MaxTimeslicesTime)
					throw gcnew InvalidOperationException(); // paranoia
				unsigned long newTime = (nextItem->HibernateTime -= EngineSettings::MaxTimeslicesTime);
                if (newTime < EngineSettings::MaxTimeslicesTime)
                {
                    //Console::WriteLine(L"Timeline: Dehibernate {" + newTime.ToString() + "}");   
                    // remove node
					LinkKind::Remove(lastItem, nextItem);
                    // add to timeline
                    AddValue(nextItem, newTime);
                }
                else
                    lastItem = nextItem;
            }
        }
    }
}}