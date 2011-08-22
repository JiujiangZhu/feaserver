#pragma once
#include "Core\CuContext.h"
using namespace System;
namespace FeaServer { namespace Engine {
	class Engine
	{
	private:
        unsigned long _timesliceIndex;
		bool _rebuildWorkingFractions;
		CuContext _context;

	public:
		Engine(CuContext context)
			: _context(context), _timesliceIndex(0) { }

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
					unsigned long lastFractionTime = 0;
					// advance time
					_rebuildWorkingFractions = true;
					time -= (long)(TimePrecision::TimeScaler - lastFractionTime);
				}
				// next slice
				_timesliceIndex++;
				if (_timesliceIndex >= EngineSettings::MaxTimeslices)
				{
					_timesliceIndex = 0;
					//DehibernateAnyValues();
				}
			}
		}

		int Foo() { return 0; }
	};

	//static Engine s_engine(s_context);
}}
