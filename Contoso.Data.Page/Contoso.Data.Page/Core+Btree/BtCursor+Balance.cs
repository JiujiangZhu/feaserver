using System.Diagnostics;
using Contoso.Sys;

namespace Contoso.Core
{
    public partial class BtCursor
    {
        static byte[] aBalanceQuickSpace = new byte[13];

        internal SQLITE balance()
        {
            var rc = SQLITE.OK;
            var nMin = (int)this.pBt.usableSize * 2 / 3;
#if DEBUG
            var balance_quick_called = 0;
            var balance_deeper_called = 0;
#else
            var balance_quick_called = 0;
            var balance_deeper_called = 0;
#endif
            do
            {
                var iPage = this.iPage;
                var pPage = this.apPage[iPage];
                if (iPage == 0)
                {
                    if (pPage.nOverflow != 0)
                    {
                        // The root page of the b-tree is overfull. In this case call the balance_deeper() function to create a new child for the root-page
                        // and copy the current contents of the root-page to it. The next iteration of the do-loop will balance the child page.
                        Debug.Assert((balance_deeper_called++) == 0);
                        rc = MemPage.balance_deeper(pPage, ref this.apPage[1]);
                        if (rc == SQLITE.OK)
                        {
                            this.iPage = 1;
                            this.aiIdx[0] = 0;
                            this.aiIdx[1] = 0;
                            Debug.Assert(this.apPage[1].nOverflow != 0);
                        }
                    }
                    else
                        break;
                }
                else if (pPage.nOverflow == 0 && pPage.nFree <= nMin)
                    break;
                else
                {
                    var pParent = this.apPage[iPage - 1];
                    var iIdx = this.aiIdx[iPage - 1];
                    rc = Pager.sqlite3PagerWrite(pParent.pDbPage);
                    if (rc == SQLITE.OK)
                    {
#if !SQLITE_OMIT_QUICKBALANCE
                        if (pPage.hasData != 0 && pPage.nOverflow == 1 && pPage.aOvfl[0].idx == pPage.nCell && pParent.pgno != 1 && pParent.nCell == iIdx )
                        {
                            // Call balance_quick() to create a new sibling of pPage on which to store the overflow cell. balance_quick() inserts a new cell
                            // into pParent, which may cause pParent overflow. If this happens, the next interation of the do-loop will balance pParent
                            // use either balance_nonroot() or balance_deeper(). Until this happens, the overflow cell is stored in the aBalanceQuickSpace[]
                            // buffer.
                            // The purpose of the following Debug.Assert() is to check that only a single call to balance_quick() is made for each call to this
                            // function. If this were not verified, a subtle bug involving reuse of the aBalanceQuickSpace[] might sneak in.
                            Debug.Assert((balance_quick_called++) == 0);
                            rc = MemPage.balance_quick(pParent, pPage, aBalanceQuickSpace);
                        }
                        else
#endif
                        {
                            // In this case, call balance_nonroot() to redistribute cells between pPage and up to 2 of its sibling pages. This involves
                            // modifying the contents of pParent, which may cause pParent to become overfull or underfull. The next iteration of the do-loop
                            // will balance the parent page to correct this.
                            // If the parent page becomes overfull, the overflow cell or cells are stored in the pSpace buffer allocated immediately below.
                            // A subsequent iteration of the do-loop will deal with this by calling balance_nonroot() (balance_deeper() may be called first,
                            // but it doesn't deal with overflow cells - just moves them to a different page). Once this subsequent call to balance_nonroot()
                            // has completed, it is safe to release the pSpace buffer used by the previous call, as the overflow cell data will have been
                            // copied either into the body of a database page or into the new pSpace buffer passed to the latter call to balance_nonroot().
                            var pSpace = new byte[this.pBt.pageSize];// u8 pSpace = sqlite3PageMalloc( pCur.pBt.pageSize );
                            rc = MemPage.balance_nonroot(pParent, iIdx, null, iPage == 1 ? 1 : 0);
                            // The pSpace buffer will be freed after the next call to balance_nonroot(), or just before this function returns, whichever comes first.
                        }
                    }
                    pPage.nOverflow = 0;
                    // The next iteration of the do-loop balances the parent page.
                    pPage.releasePage();
                    this.iPage--;
                }
            } while (rc == SQLITE.OK);
            return rc;
        }
    }
}
