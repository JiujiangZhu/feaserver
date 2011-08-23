#pragma once
#include "Core\CalContext.h"
#include "Engine.h"
using namespace System;

namespace FeaServer { namespace Engine {
	using namespace Core;
	public ref class CalEngine : IEngine
	{
	private:
		ObjectCollection^ _objects;
		TypeCollection^ _types;

	public:
		CalEngine()
			: _objects(nullptr), _types(nullptr) {
			if (!_context.Initialize())
				throw gcnew Exception(L"Test");
		}
		~CalEngine()
		{
			_context.Dispose();
		}

#pragma region Engine
		property ObjectCollection^ Objects
		{
			virtual ObjectCollection^ get() { return _objects; }
		}

		property TypeCollection^ Types
		{
			virtual TypeCollection^ get() { return _types; }
		}
   
		virtual void EvaluateFrame(Int64 time)
		{
			s_engine.EvaluateFrame(time);
		}
#pragma endregion

	};
}}
