using System.Diagnostics;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using Pgno = System.UInt32;
using Contoso.Collections;

namespace Contoso.Core
{
    public partial class BtShared
    {
        internal RC btreeSetHasContent(Pgno pgno)
        {
            var rc = RC.OK;
            if (this.HasContent == null)
            {
                Debug.Assert(pgno <= this.Pages);
                this.HasContent = new BitArray(this.Pages);
            }
            if (rc == RC.OK && pgno <= this.HasContent.Length)
                rc = this.HasContent.Set(pgno);
            return rc;
        }

        internal bool btreeGetHasContent(Pgno pgno)
        {
            var p = this.HasContent;
            return (p != null && (pgno > p.Length || p.Get(pgno) != 0));
        }

        internal void btreeClearHasContent()
        {
            BitArray.Destroy(ref this.HasContent);
            this.HasContent = null;
        }

#if DEBUG
        internal int countWriteCursors()
        {
            var r = 0;
            for (var pCur = this.Cursors; pCur != null; pCur = pCur.Next)
                if (pCur.Writeable && pCur.State != CURSOR.FAULT)
                    r++;
            return r;
        }
#else
        internal int countWriteCursors() { return -1; }
#endif
    }
}
