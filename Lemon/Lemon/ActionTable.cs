using System;
using System.Diagnostics;

namespace Lemon
{
    public class ActionTable
    {
        public struct lookahead_action
        {
            public int lookahead;
            public int action;
        }
        public int nAction;
        public int nActionAlloc;
        public lookahead_action[] aAction;
        public lookahead_action[] aLookahead;
        public int mnLookahead;
        public int mnAction;
        public int mxLookahead;
        public int nLookahead;
        public int nLookaheadAlloc;

        public void Action(int lookahead, int action)
        {
            if (nLookahead >= nLookaheadAlloc)
            {
                nLookaheadAlloc += 25;
                Array.Resize(ref aLookahead, nLookaheadAlloc);
            }
            if (nLookahead == 0)
            {
                mxLookahead = lookahead;
                mnLookahead = lookahead;
                mnAction = action;
            }
            else
            {
                if (mxLookahead < lookahead)
                    mxLookahead = lookahead;
                if (mnLookahead > lookahead)
                {
                    mnLookahead = lookahead;
                    mnAction = action;
                }
            }
            aLookahead[nLookahead].lookahead = lookahead;
            aLookahead[nLookahead].action = action;
            nLookahead++;
        }
        
        /* Return the number of entries in the yy_action table */
        public int Size
        {
            get { return nAction; }
        }

        /* The value for the N-th entry in yy_action */
        public int GetAction(int index)
        {
            return aAction[index].action;
        }

        /* The value for the N-th entry in yy_lookahead */
        public int GetLookahead(int index)
        {
            return aAction[index].lookahead;
        }

        public int Insert()
        {
            Debug.Assert(nLookahead > 0);

            /* Make sure we have enough space to hold the expanded action table in the worst case.  The worst case occurs if the transaction set must be appended to the current action table */
            var n = mxLookahead + 1;
            if (nAction + n >= nActionAlloc)
            {
                var oldAlloc = nActionAlloc;
                nActionAlloc = nAction + n + nActionAlloc + 20;
                Array.Resize(ref aAction, nActionAlloc);
                for (var index = oldAlloc; index < nActionAlloc; index++)
                {
                    aAction[index].lookahead = -1;
                    aAction[index].action = -1;
                }
            }

            /* Scan the existing action table looking for an offset that is a duplicate of the current transaction set.  Fall out of the loop if and when the duplicate is found.
            ** i is the index in aAction[] where mnLookahead is inserted. */
            var i = nAction - 1;
            for (; i >= 0; i--)
                if (aAction[i].lookahead == mnLookahead)
                {
                    /* All lookaheads and actions in the aLookahead[] transaction must match against the candidate aAction[i] entry. */
                    if (aAction[i].action != mnAction)
                        continue;
                    var j = 0;
                    for (; j < nLookahead; j++)
                    {
                        var k = aLookahead[j].lookahead - mnLookahead + i;
                        if ((k < 0) || (k >= nAction))
                            break;
                        if (aLookahead[j].lookahead != aAction[k].lookahead)
                            break;
                        if (aLookahead[j].action != aAction[k].action)
                            break;
                    }
                    if (j < nLookahead)
                        continue;

                    /* No possible lookahead value that is not in the aLookahead[] transaction is allowed to match aAction[i] */
                    var n2 = 0;
                    for (var j2 = 0; j2 < nAction; j2++)
                    {
                        if (aAction[j2].lookahead < 0)
                            continue;
                        if (aAction[j2].lookahead == j2 + mnLookahead - i)
                            n2++;
                    }
                    if (n2 == nLookahead)
                        break;  /* An exact match is found at offset i */
                }

            /* If no existing offsets exactly match the current transaction, find an an empty offset in the aAction[] table in which we can add the aLookahead[] transaction. */
            if (i < 0)
            {
                /* Look for holes in the aAction[] table that fit the current aLookahead[] transaction.  Leave i set to the offset of the hole.
                ** If no holes are found, i is left at p->nAction, which means the transaction will be appended. */
                for (i = 0; i < nActionAlloc - mxLookahead; i++)
                    if (aAction[i].lookahead < 0)
                    {
                        var j = 0;
                        for (; j < nLookahead; j++)
                        {
                            var k = aLookahead[j].lookahead - mnLookahead + i;
                            if (k < 0)
                                break;
                            if (aAction[k].lookahead >= 0)
                                break;
                        }
                        if (j < nLookahead)
                            continue;
                        for (j = 0; j < nAction; j++)
                            if (aAction[j].lookahead == j + mnLookahead - i)
                                break;
                        if (j == nAction)
                            break;  /* Fits in empty slots */
                    }
            }
            /* Insert transaction set at index i. */
            for (var j = 0; j < nLookahead; j++)
            {
                var k = aLookahead[j].lookahead - mnLookahead + i;
                aAction[k] = aLookahead[j];
                if (k >= nAction)
                    nAction = k + 1;
            }
            nLookahead = 0;

            /* Return the offset that is added to the lookahead in order to get the index into yy_action of the action */
            return i - mnLookahead;
        }
    }
}
