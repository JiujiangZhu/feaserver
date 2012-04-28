using System;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // pArray is a pointer to an array of objects.  Each object in the array is szEntry bytes in size.  This routine allocates a new
        // object on the end of the array.
        //
        // pnEntry is the number of entries already in use.  pnAlloc is the previously allocated size of the array.  initSize is the
        // suggested initial array size allocation.
        //
        // The index of the new entry is returned in pIdx.
        //
        // This routine returns a pointer to the array of objects.  This might be the same as the pArray parameter or it might be a different
        // pointer if the array was resized.
        internal static T[] sqlite3ArrayAllocate<T>(sqlite3 db, T[] pArray, int szEntry, int initSize, ref int pnEntry, ref int pnAlloc, ref int pIdx) where T : new()
        {
            if (pnEntry >= pnAlloc)
            {
                int newSize;
                newSize = (pnAlloc) * 2 + initSize;
                pnAlloc = newSize;
                Array.Resize(ref pArray, newSize);
            }
            pArray[pnEntry] = new T();
            pIdx = pnEntry;
            ++pnEntry;
            return pArray;
        }

        // Append a new element to the given IdList.  Create a new IdList if need be.
        // A new IdList is returned, or NULL if malloc() fails.
        internal static IdList sqlite3IdListAppend(sqlite3 db, int null_2, Token pToken) { return sqlite3IdListAppend(db, null, pToken); }
        internal static IdList sqlite3IdListAppend(sqlite3 db, IdList pList, Token pToken)
        {
            int i = 0;
            if (pList == null)
            {
                pList = new IdList();
                if (pList == null)
                    return null;
                pList.nAlloc = 0;
            }
            pList.a = (IdList_item[])sqlite3ArrayAllocate(db, pList.a, -1, 5, ref pList.nId, ref pList.nAlloc, ref i);
            if (i < 0)
            {
                sqlite3IdListDelete(db, ref pList);
                return null;
            }
            pList.a[i].zName = sqlite3NameFromToken(db, pToken);
            return pList;
        }

        // Delete an IdList.
        internal static void sqlite3IdListDelete(sqlite3 db, ref IdList pList)
        {
            if (pList == null)
                return;
            for (var i = 0; i < pList.nId; i++)
                sqlite3DbFree(db, ref pList.a[i].zName);
            sqlite3DbFree(db, ref pList.a);
            sqlite3DbFree(db, ref pList);
        }

        // Return the index in pList of the identifier named zId.  Return -1 if not found.
        internal static int sqlite3IdListIndex(IdList pList, string zName)
        {
            if (pList == null)
                return -1;
            for (var i = 0; i < pList.nId; i++)
                if (pList.a[i].zName.Equals(zName, StringComparison.InvariantCultureIgnoreCase))
                    return i;
            return -1;
        }
    }
}