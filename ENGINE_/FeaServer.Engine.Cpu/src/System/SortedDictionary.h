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

namespace System {
	template <typename TKey, typename TValue>
	struct KeyValuePair
	{
		TKey key;
		TValue value;
	};

	template <typename TKey, typename TValue>
	class SortedDictionary
	{
	protected:
		fallocDeviceContext* _deviceCtx;
		TreeSet<KeyValuePair<TKey, TValue>> _set;

	public:
		//__device__ SortedDictionary(fallocDeviceContext* deviceCtx)
		//	: _deviceCtx(deviceCtx) { }
		__device__ static SortedDictionary* ctor(fallocDeviceContext* deviceCtx)
		{
			SortedDictionary* dictionary = (SortedDictionary*)falloc(deviceCtx, sizeof(SortedDictionary));
			dictionary->_deviceCtx = dictionary->_set._deviceCtx = deviceCtx;
			return dictionary;
		}

	protected:
		__device__ bool TryGetValue(TKey key, TValue* value)
		{
			KeyValuePair<TKey, TValue> pair;
			pair.key = key; //memcpy(&pair.key, &key, sizeof(key));
			//x* node = _set.FindNode(pair);
			return false;
		}

		__device__ void Add(TKey key, TValue value)
		{
			KeyValuePair<TKey, TValue>* pair = (KeyValuePair<TKey, TValue>*)falloc(_deviceCtx, sizeof(KeyValuePair<TKey, TValue>));
			pair.key = key; //memcpy(&pair.key, &key, sizeof(key));
			_set.Add(pair);
		}
	};
}