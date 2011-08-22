#include "Engine.h"
namespace TimeServices { namespace Engine {
	void Engine::AddValue(LinkKind* value, unsigned long time)
	{
		//Console::WriteLine(L"Timeline: Add {" + time.ToString() + L"}");
		unsigned long slice = (time >> TimePrecision::TimePrecisionBits);
		unsigned long fraction = (time & TimePrecision::TimePrecisionMask);
		if (slice < EngineSettings::MaxTimeslices)
		{
			// time is fractional only
			// enhance: could check for existance in hash
			if ((slice == 0) && (fraction < _maxWorkingFraction))
				_rebuildWorkingFractions = true;
			// roll timeslice for index
			slice += _sliceIndex;
			if (slice >= EngineSettings::MaxTimeslices)
				slice -= EngineSettings::MaxTimeslices;
			SliceFractions* fractions = &_slices[slice].Fractions;
			// add to list
			SliceNode* node = nullptr;
			if (!fractions->TryGetValue(fraction, node))
			{
				value->NextLink = nullptr;
				fractions->Add(fraction, value);
			}
			else
				LinkKind::AddFirst(node->Chain, value);
			return;
		}
		HibernateValue(value, time);
	}

    void Engine::AddValue(ListKind* value, unsigned long time)
    {
        //Console::WriteLine(L"Timeline: Add {" + time.ToString() + L"}");
		unsigned long slice = (time >> TimePrecision::TimePrecisionBits);
		unsigned long fraction = (time & TimePrecision::TimePrecisionMask);
		if (slice < EngineSettings::MaxTimeslices)
        {
            // time is fractional only
            // enhance: could check for existance in hash
            if ((slice == 0) && (fraction < _maxWorkingFraction))
                _rebuildWorkingFractions = true;
            // roll timeslice for index
            slice += _sliceIndex;
			if (slice >= EngineSettings::MaxTimeslices)
				slice -= EngineSettings::MaxTimeslices;
            SliceFractions* fractions = &_slices[slice].Fractions;
            // add to list
            SliceNode* node = nullptr;
            if (!fractions->TryGetValue(fraction, node))
                fractions->Add(fraction, value);
            else
            {
				ListKind::List* list = node->List;
                if (list == nullptr)
                    list = node->List = new ListKind::List();
                list->Add(value);
            }
            return;
        }
        HibernateValue(value, time);
    }
}}