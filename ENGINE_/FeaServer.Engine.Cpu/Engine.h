#pragma once
#include "Core\CpuContext.h"
#include "Core\LinkKind.h"
#include "Core\ListKind.h"
#include "Core\Slice.h"
using namespace System;
namespace TimeServices { namespace Engine {
	using namespace Core;

	class Engine
	{
	private:
		CpuContext _context;
        unsigned long _sliceIndex;
		Slice _slices[EngineSettings::MaxTimeslices];
		unsigned long _workingFractions[EngineSettings::MaxWorkingFractions];
        unsigned long _minWorkingFraction; //: (sorted)dictionary removed are expensive. virtually remove with a window (_minWorkingFraction).
        unsigned long _maxWorkingFraction;
		bool _rebuildWorkingFractions;

	public:
		Engine(CpuContext context)
			: _context(context), _sliceIndex(0) {
            //for (int sliceIndex = 0; sliceIndex < EngineSettings::MaxTimeslices; sliceIndex++)
            //    _slices[timesliceIndex].Fractions = gcnew SortedDictionary<unsigned long, SliceNode^>();
		}

		void EvaluateFrame(__int64 time)
		{
			while (true)
			{
				// exit if frame completed
				if (time <= 0)
					return;
				if (Foo() == 0)
				{
					// no fractions available, advance a whole time
					_rebuildWorkingFractions = true;
					time -= (long)TimePrecision::TimeScaler;
				} else {
					unsigned long lastFraction = 0;
					// advance time
					_rebuildWorkingFractions = true;
					time -= (long)(TimePrecision::TimeScaler - lastFraction);
				}
				// next slice
				_sliceIndex++;
				if (_sliceIndex >= EngineSettings::MaxTimeslices)
				{
					_sliceIndex = 0;
					DehibernateAnyValues();
				}
			}
		}

		int Foo() { return 0; }
	private:
		void AddValue(LinkKind* value, unsigned long time);
		void AddValue(ListKind* value, unsigned long time);
		void HibernateValue(LinkKind* value, unsigned long time);
		void HibernateValue(ListKind* value, unsigned long time);
		void DehibernateAnyValues();

	private:
		struct HibernateState
		{
            unsigned long Time;
            ListKind* Object;
		public:
			struct List
			{
			public:
				void Add(ListKind* value)
				{
				}
			};
		};
		struct HibernateSegment
		{
			HibernateState::List* List;
			LinkKind* Chain;
		};
		HibernateSegment _hibernateSegments[EngineSettings::MaxHibernateSegments];
	};

	static Engine s_engine(s_context);
}}
