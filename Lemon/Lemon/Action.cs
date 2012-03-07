using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class Action
    {
        public Symbol Symbol;
        public ActionType Type;
        public State State;
        public Rule Rule;
        //public Action Next;
        public Action Collide;

        public class KeyComparer : IComparer<Action>
        {
            public int Compare(Action x, Action y)
            {
                var v = x.Symbol.ID - y.Symbol.ID;
                if (v == 0)
                    v = (int)x.Type - (int)y.Type;
                if (v == 0 && x.Type == ActionType.Reduce)
                    v = x.Rule.ID - y.Rule.ID;
                if (v == 0)
                    v = (object.ReferenceEquals(y, x) ? 0 : -1);
                return v;
            }
        }

        public static int ResolveConflict(Action x, Action y, Symbol errorSymbol)
        {
            int errors = 0;
            Debug.Assert(object.ReferenceEquals(x.Symbol, y.Symbol)); /* Otherwise there would be no conflict */
            if (x.Type == ActionType.Shift && y.Type == ActionType.Shift)
            {
                y.Type = ActionType.SSConflict;
                errors++;
            }
            if (x.Type == ActionType.Shift && y.Type == ActionType.Reduce)
            {
                var x2 = x.Symbol;
                var y2 = y.Rule.PrecedenceSymbol;
                /* Not enough precedence information. */
                if (y2 == null || x2.Precedence < 0 || y2.Precedence < 0)
                {
                    y.Type = ActionType.SRConflict;
                    errors++;
                }
                /* higher precedence wins */
                else if (x2.Precedence > y2.Precedence)
                    y.Type = ActionType.RDResolved;
                else if (x2.Precedence < y2.Precedence)
                    x.Type = ActionType.SHResolved;
                /* Use operator */
                else if (x2.Precedence == y2.Precedence && x2.Association == Association.Right)
                    y.Type = ActionType.RDResolved; /* associativity */
                /* to break tie */
                else if (x2.Precedence == y2.Precedence && x2.Association == Association.Left)
                    x.Type = ActionType.SHResolved;
                else
                {
                    y.Type = ActionType.SRConflict;
                    errors++;
                }
            }
            else if (x.Type == ActionType.Reduce && y.Type == ActionType.Reduce)
            {
                var x2 = x.Rule.PrecedenceSymbol;
                var y2 = y.Rule.PrecedenceSymbol;
                if (x2 == null || y2 == null || x2.Precedence < 0 || y2.Precedence < 0 || x2.Precedence == y2.Precedence)
                {
                    y.Type = ActionType.RRConflict;
                    errors++;
                }
                else if (x2.Precedence > y2.Precedence)
                    y.Type = ActionType.RDResolved;
                else if (x2.Precedence < y2.Precedence)
                    x.Type = ActionType.RDResolved;
            }
            /* The REDUCE/SHIFT case cannot happen because SHIFTs come before REDUCEs on the list.  If we reach this point it must be because the parser conflict had already been resolved. */
            else
                Debug.Assert(
                    x.Type == ActionType.SHResolved || x.Type == ActionType.RDResolved || x.Type == ActionType.SSConflict || x.Type == ActionType.SRConflict || x.Type == ActionType.RRConflict ||
                    y.Type == ActionType.SHResolved || y.Type == ActionType.RDResolved || y.Type == ActionType.SSConflict || y.Type == ActionType.SRConflict || y.Type == ActionType.RRConflict
                );
            return errors;
        }

        internal int Compute(Context ctx)
        {
            switch (Type)
            {
                case ActionType.Shift: return State.ID;
                case ActionType.Reduce: return Rule.ID + ctx.States;
                case ActionType.Error: return ctx.States + ctx.Rules;
                case ActionType.Accept: return ctx.States + ctx.Rules + 1;
                default: return -1;
            }
        }
    }
}
