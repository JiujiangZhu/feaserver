using System;
namespace Contoso.Core
{
    public class HashEx
    {
        public class _ht
        {            
            public int count;
            public HashElem chain;
        }

        public class HashElem
        {
            public HashElem next;
            public HashElem prev;
            public object data;
            public string pKey;
            public int nKey;
        }

        public uint htsize = 31;
        public uint count; 
        public HashElem first;
        public _ht[] ht;

        public HashEx Clone() { return (HashEx)MemberwiseClone(); }
    }
}
