#pragma once
namespace System {
	template <typename TKey, typename TValue>
	class SortedDictionary
	{
	protected:
		__device__ bool TryGetValue(TKey key, TValue* value)
		{
			return false;
		}

		__device__ void Add(TKey key, TValue value)
		{
		}
	};
}