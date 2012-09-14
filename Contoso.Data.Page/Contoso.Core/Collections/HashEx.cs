namespace Contoso.Collections
{
    public class HashEx
    {
        private class _ht
        {            
            public int Count;
            public HashElem Chain;
        }

        private class HashElem
        {
            public HashElem next;
            public HashElem prev;
            public object data;
            public string pKey;
            public int nKey;
        }

        private uint htsize = 31;
        private uint count;
        private HashElem first;
        private _ht[] ht;

        private HashEx Clone() { return (HashEx)MemberwiseClone(); }
    }
}
