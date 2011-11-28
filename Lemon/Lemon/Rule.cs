using System;

namespace Lemon
{
    public class Rule
    {
        public Symbol lhs;
        public string lhsalias;
        public int lhsStart;
        public int ruleline;
        public int nrhs;
        public Symbol[] rhs;
        public string[] rhsalias;
        public int line;
        public string code;
        public Symbol precsym;
        public int index;
        public bool canReduce;
        public Rule nextlhs;
        public Rule next;
    }
}
