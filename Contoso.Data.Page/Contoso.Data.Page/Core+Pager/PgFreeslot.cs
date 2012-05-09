using System;
namespace Contoso.Core
{
    public class PgFreeslot
    {
        public PgFreeslot pNext;    // Next free slot
        public PgHdr _PgHdr;        // Next Free Header
    }
}
