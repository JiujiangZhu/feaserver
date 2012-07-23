namespace Contoso.Core
{
    public interface IVdbe
    {
        Btree.UnpackedRecord sqlite3VdbeRecordUnpack(KeyInfo keyInfo, int nKey, byte[] pKey, Btree.UnpackedRecord aSpace, int count);
        void sqlite3VdbeDeleteUnpackedRecord(Btree.UnpackedRecord r);
        int sqlite3VdbeRecordCompare(int nCell, byte[] pCellKey, Btree.UnpackedRecord pIdxKey);
        int sqlite3VdbeRecordCompare(int nCell, byte[] p, int p_2, Btree.UnpackedRecord pIdxKey);
    }
}
