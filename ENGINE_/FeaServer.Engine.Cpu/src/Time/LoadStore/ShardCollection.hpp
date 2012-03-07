#pragma region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma endregion
#pragma once
#include "Shard.hpp"
#include "..\Scheduler\SliceCollection.hpp"

namespace Time { namespace LoadStore {
#ifdef _SHARDCOLLECTION

#else
	#define SHARDCOLLECTION

	typedef struct { ulong key; Shard* value; } ShardPair;
	class ShardCollection
	{
	private:
		System::TreeSet<ShardPair> _set;
		fallocContext* _fallocCtx;

		__device__ bool TryGetValue(ulong key, Shard** value)
		{
			ShardPair pair;
			pair.key = key;
			System::Node<ShardPair>* node = _set.FindNode(pair);
			if (node == nullptr)
				return false;
			*value = node->item.value;
			return true;
		}

		__device__ void Add(ulong key, Shard* value)
		{
			ShardPair pair;
			pair.key = key; pair.value = value;
			_set.Add(pair);
		}

	public:
		__device__ void xtor(fallocContext* fallocCtx)
		{
			trace(ShardCollection, "xtor");
			_fallocCtx = fallocCtx;
			_set.xtor(0, fallocCtx);
		}
		__device__ void Dispose()
		{
			trace(ShardCollection, "Dispose");
		}

		__device__ void Load(Scheduler::SliceCollection* slices, ulong shard)
        {
			trace(ShardCollection, "Load %d", shard);
            Shard* shardAsObject;
            if (!TryGetValue(shard, &shardAsObject))
			{
				shardAsObject = falloc<Shard>(_fallocCtx);
				shardAsObject->xtor(_fallocCtx);
                Add(shard, shardAsObject);
			}
            //shardAsObject->Elements.Add(element, 0);
			//slices->Schedule(nullptr, 0);
        }

	};

#endif
}}
