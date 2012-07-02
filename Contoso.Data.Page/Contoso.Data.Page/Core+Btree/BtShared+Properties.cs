﻿using System.Diagnostics;
using Contoso.Sys;
using CURSOR = Contoso.Core.BtreeCursor.CursorState;
using Pgno = System.UInt32;

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
                this.HasContent = new Bitvec(this.Pages);
            }
            if (rc == RC.OK && pgno <= this.HasContent.sqlite3BitvecSize())
                rc = this.HasContent.sqlite3BitvecSet(pgno);
            return rc;
        }

        internal bool btreeGetHasContent(Pgno pgno)
        {
            var p = this.HasContent;
            return (p != null && (pgno > p.sqlite3BitvecSize() || p.sqlite3BitvecTest(pgno) != 0));
        }

        internal void btreeClearHasContent()
        {
            Bitvec.sqlite3BitvecDestroy(ref this.HasContent);
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
