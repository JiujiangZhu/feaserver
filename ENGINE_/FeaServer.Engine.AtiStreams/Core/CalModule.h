#pragma once
#include "cal.h"
namespace TimeServices { namespace Engine { namespace Core {
	class CalModule
	{
	private:
		CALmodule _module;
		CalContext* _context;

	public:
		CalModule::CalModule(CalContext& context)
			: _context(&context), _module(0) { }
		bool Initialize(const CALchar* ilKernel, bool disassemble);
	};
}}}
