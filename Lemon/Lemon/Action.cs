using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class Action : IComparer<Action>
    {
        public Symbol sp;
        public ActionType type;
        public State stp;
        public Rule rp;
        public Action next;
        public Action collide;

        public Action(ref Action actionHead)
        {
            next = actionHead;
            actionHead = this;
        }

        public static Action Sort(Action action)
        {
            //var ap = msort((char *)ap,(char **)&ap->next, (int(*)(const char*,const char*))actioncmp);
            //return ap;
            return null;
        }

        public int Compare(Action x, Action y)
        {
            var rc = x.sp.Index - y.sp.Index;
            if (rc == 0)
                rc = (int)x.type - (int)y.type;
            if ((rc == 0) && (x.type == ActionType.Reduce))
                rc = x.rp.index - y.rp.index;
            //if (rc == 0)
            //    rc = (int)(y.Equals(x));
            return rc;
        }

        public static int ResolveConflict(Action apx, Action apy, Symbol errsym)
        {
            //  struct symbol *spx, *spy;
            int errcnt = 0;
            //  assert( apx->sp==apy->sp );  /* Otherwise there would be no conflict */
            if ((apx.type == ActionType.Shift) && (apy.type == ActionType.Shift))
            {
                apy.type = ActionType.SSConflict;
                errcnt++;
            }
            if ((apx.type == ActionType.Shift) && (apy.type == ActionType.Reduce))
            {
                var spx = apx.sp;
                var spy = apy.rp.precsym;
                /* Not enough precedence information. */
                if ((spy == null) || (spx.Precedence < 0) || (spy.Precedence < 0))
                {

                    apy.type = ActionType.SRConflict;
                    errcnt++;
                }
                /* higher precedence wins */
                else if (spx.Precedence > spy.Precedence)
                    apy.type = ActionType.RDResolved;
                else if (spx.Precedence < spy.Precedence)
                    apx.type = ActionType.SHResolved;
                /* Use operator */
                else if ((spx.Precedence == spy.Precedence) && (spx.Association == Association.Right))
                    apy.type = ActionType.RDResolved; /* associativity */
                /* to break tie */
                else if ((spx.Precedence == spy.Precedence) && (spx.Association == Association.Left))
                    apx.type = ActionType.SHResolved;
                else
                {
                    apy.type = ActionType.SRConflict;
                    errcnt++;
                }
            }
            else if ((apx.type == ActionType.Reduce) && (apy.type == ActionType.Reduce))
            {
                var spx = apx.rp.precsym;
                var spy = apy.rp.precsym;
                if ((spx == null) || (spy == null) || (spx.Precedence < 0) || (spy.Precedence < 0) || (spx.Precedence == spy.Precedence))
                {
                    apy.type = ActionType.RRConflict;
                    errcnt++;
                }
                else if (spx.Precedence > spy.Precedence)
                    apy.type = ActionType.RDResolved;
                else if (spx.Precedence < spy.Precedence)
                    apx.type = ActionType.RDResolved;
            }
            /* The REDUCE/SHIFT case cannot happen because SHIFTs come before REDUCEs on the list.  If we reach this point it must be because the parser conflict had already been resolved. */
            else
                Debug.Assert(
                  (apx.type == ActionType.SHResolved) ||
                  (apx.type == ActionType.RDResolved) ||
                  (apx.type == ActionType.SSConflict) ||
                  (apx.type == ActionType.SRConflict) ||
                  (apx.type == ActionType.RRConflict) ||
                  (apy.type == ActionType.SHResolved) ||
                  (apy.type == ActionType.RDResolved) ||
                  (apy.type == ActionType.SSConflict) ||
                  (apy.type == ActionType.SRConflict) ||
                  (apy.type == ActionType.RRConflict)
                );
            return errcnt;
        }

        internal int ComputeAaction(Lemon lemon)
        {
            switch (type)
            {
                case ActionType.Shift: return stp.statenum;
                case ActionType.Reduce: return rp.index + lemon.nstate;
                case ActionType.Error: return lemon.nstate + lemon.nrule;
                case ActionType.Accept: return lemon.nstate + lemon.nrule + 1;
                default: return -1;
            }
        }
    }
}
