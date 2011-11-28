using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class State : IComparer<State>
    {
        public static StateCollection States = new StateCollection();

        public Config bp;
        public Config cfp;
        public int statenum;
        public Action ap;
        public int nTknAct;
        public int nNtAct;
        public int iTknOfst;
        public int iNtOfst;
        public int iDflt;

        public int Compare(State x, State y)
        {
            var n = y.nNtAct - x.nNtAct;
            if (n == 0)
            {
                n = y.nTknAct - x.nTknAct;
                if (n == 0)
                    n = y.statenum - x.statenum;
            }
            Debug.Assert(n != 0);
            return n;
        }
    }

    //    static int stateResortCompare(const void *a, const void *b){
    //  const struct state *pA = *(const struct state**)a;
    //  const struct state *pB = *(const struct state**)b;
    //  int n;

    //  n = pB->nNtAct - pA->nNtAct;
    //  if( n==0 ){
    //    n = pB->nTknAct - pA->nTknAct;
    //    if( n==0 ){
    //      n = pB->statenum - pA->statenum;
    //    }
    //  }
    //  assert( n!=0 );
    //  return n;
    //}

}
