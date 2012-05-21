using System;
using System.Diagnostics;

namespace Lemon
{
    public partial class Emitter
    {
        public class ActionTable
        {
            public struct lookahead_action
            {
                public int Lookahead;
                public int Action;
            }

            public int _UsedActions;
            public int _AllocatedActions;
            public lookahead_action[] Actions;
            public lookahead_action[] Lookaheads;
            public int mnLookahead;
            public int mnAction;
            public int mxLookahead;
            public int _UsedLookaheads;
            public int _AllocatedLookaheads;

            public void Action(int lookahead, int actionID)
            {
                if (_UsedLookaheads >= _AllocatedLookaheads)
                {
                    _AllocatedLookaheads += 25;
                    Array.Resize(ref Lookaheads, _AllocatedLookaheads);
                }
                if (_UsedLookaheads == 0)
                {
                    mxLookahead = lookahead;
                    mnLookahead = lookahead;
                    mnAction = actionID;
                }
                else
                {
                    if (mxLookahead < lookahead)
                        mxLookahead = lookahead;
                    if (mnLookahead > lookahead)
                    {
                        mnLookahead = lookahead;
                        mnAction = actionID;
                    }
                }
                Lookaheads[_UsedLookaheads].Lookahead = lookahead;
                Lookaheads[_UsedLookaheads].Action = actionID;
                _UsedLookaheads++;
            }

            /* Return the number of entries in the yy_action table */
            public int Size
            {
                get { return _UsedActions; }
            }

            /* The value for the N-th entry in yy_action */
            public int GetAction(int index)
            {
                return Actions[index].Action;
            }

            /* The value for the N-th entry in yy_lookahead */
            public int GetLookahead(int index)
            {
                return Actions[index].Lookahead;
            }

            public int Insert()
            {
                Debug.Assert(_UsedLookaheads > 0);

                /* Make sure we have enough space to hold the expanded action table in the worst case.  The worst case occurs if the transaction set must be appended to the current action table */
                var n = mxLookahead + 1;
                if (_UsedActions + n >= _AllocatedActions)
                {
                    var oldAlloc = _AllocatedActions;
                    _AllocatedActions = _UsedActions + n + _AllocatedActions + 20;
                    Array.Resize(ref Actions, _AllocatedActions);
                    for (var index = oldAlloc; index < _AllocatedActions; index++)
                    {
                        Actions[index].Lookahead = -1;
                        Actions[index].Action = -1;
                    }
                }

                /* Scan the existing action table looking for an offset that is a duplicate of the current transaction set.  Fall out of the loop if and when the duplicate is found.
                ** i is the index in aAction[] where mnLookahead is inserted. */
                var i = _UsedActions - 1;
                for (; i >= 0; i--)
                    if (Actions[i].Lookahead == mnLookahead)
                    {
                        /* All lookaheads and actions in the aLookahead[] transaction must match against the candidate aAction[i] entry. */
                        if (Actions[i].Action != mnAction)
                            continue;
                        var j = 0;
                        for (; j < _UsedLookaheads; j++)
                        {
                            var k = Lookaheads[j].Lookahead - mnLookahead + i;
                            if ((k < 0) || (k >= _UsedActions))
                                break;
                            if (Lookaheads[j].Lookahead != Actions[k].Lookahead)
                                break;
                            if (Lookaheads[j].Action != Actions[k].Action)
                                break;
                        }
                        if (j < _UsedLookaheads)
                            continue;

                        /* No possible lookahead value that is not in the aLookahead[] transaction is allowed to match aAction[i] */
                        var n2 = 0;
                        for (var j2 = 0; j2 < _UsedActions; j2++)
                        {
                            if (Actions[j2].Lookahead < 0)
                                continue;
                            if (Actions[j2].Lookahead == j2 + mnLookahead - i)
                                n2++;
                        }
                        if (n2 == _UsedLookaheads)
                            break;  /* An exact match is found at offset i */
                    }

                /* If no existing offsets exactly match the current transaction, find an an empty offset in the aAction[] table in which we can add the aLookahead[] transaction. */
                if (i < 0)
                {
                    /* Look for holes in the aAction[] table that fit the current aLookahead[] transaction.  Leave i set to the offset of the hole.
                    ** If no holes are found, i is left at p->nAction, which means the transaction will be appended. */
                    for (i = 0; i < _AllocatedActions - mxLookahead; i++)
                        if (Actions[i].Lookahead < 0)
                        {
                            var j = 0;
                            for (; j < _UsedLookaheads; j++)
                            {
                                var k = Lookaheads[j].Lookahead - mnLookahead + i;
                                if (k < 0)
                                    break;
                                if (Actions[k].Lookahead >= 0)
                                    break;
                            }
                            if (j < _UsedLookaheads)
                                continue;
                            for (j = 0; j < _UsedActions; j++)
                                if (Actions[j].Lookahead == j + mnLookahead - i)
                                    break;
                            if (j == _UsedActions)
                                break;  /* Fits in empty slots */
                        }
                }
                /* Insert transaction set at index i. */
                for (var j = 0; j < _UsedLookaheads; j++)
                {
                    var k = Lookaheads[j].Lookahead - mnLookahead + i;
                    Actions[k] = Lookaheads[j];
                    if (k >= _UsedActions)
                        _UsedActions = k + 1;
                }
                _UsedLookaheads = 0;

                /* Return the offset that is added to the lookahead in order to get the index into yy_action of the action */
                return i - mnLookahead;
            }
        }
    }
}
