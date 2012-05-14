using Pgno = System.UInt32;
namespace Contoso.Core
{
    public class PagerSavepoint
    {
        public long iOffset;            // Starting offset in main journal
        public long iHdrOffset;         // See above
        public Bitvec pInSavepoint;     // Set of pages in this savepoint
        public Pgno nOrig;              // Original number of pages in file
        public Pgno iSubRec;            // Index of first record in sub-journal
#if !SQLITE_OMIT_WAL
        public uint aWalData[WAL_SAVEPOINT_NDATA];        // WAL savepoint context
#else
        public object aWalData = null;      // Used for C# convenience
#endif

        public static implicit operator bool(PagerSavepoint b)
        {
            return (b != null);
        }
    }
}
