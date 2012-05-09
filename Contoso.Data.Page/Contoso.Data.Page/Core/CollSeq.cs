namespace Contoso.Core
{
    public class CollSeq
    {
        public delegate void dxDelCollSeq(ref object pDelArg);
        public delegate int dxCompare(object pCompareArg, int size1, string Key1, int size2, string Key2);

        public enum COLL
        {
            BINARY = 1,     // The default memcmp() collating sequence
            NOCASE = 2,     // The built-in NOCASE collating sequence
            REVERSE = 3,    // The built-in REVERSE collating sequence
            USER = 0,       // Any other user-defined collating sequence
        }

        public string zName;        // Name of the collating sequence, UTF-8 encoded
        public byte enc;            // Text encoding handled by xCmp()
        public COLL type;
        public object pUser;        // First argument to xCmp()
        public dxCompare xCmp;
        public dxDelCollSeq xDel;

        public CollSeq Copy()
        {
            if (this == null)
                return null;
            else
            {
                var cp = (CollSeq)MemberwiseClone();
                return cp;
            }
        }
    }
}
