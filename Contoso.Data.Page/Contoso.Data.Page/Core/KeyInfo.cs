namespace Contoso.Core
{
    public class KeyInfo
    {
        public sqlite3 db;          // The database connection
        public byte enc;            // Text encoding - one of the SQLITE_UTF* values
        public ushort nField;       // Number of entries in aColl[]
        public byte[] aSortOrder;   // Sort order for each column.  May be NULL
        public CollSeq[] aColl = new CollSeq[1];  // Collating sequence for each term of the key

        public KeyInfo Copy()
        {
            return (KeyInfo)MemberwiseClone();
        }
    }
}
