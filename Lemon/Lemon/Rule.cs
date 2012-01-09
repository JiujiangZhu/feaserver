using System;

namespace Lemon
{
    public class Rule
    {
        public Symbol LHSymbol;
        public string LHSymbolAlias;
        public int LhSymbolStart;
        public int RuleLineno;
        //public int nrhs;
        public Symbol[] RHSymbols;
        public string[] RHSymbolsAlias;
        public int Lineno;
        public string Code;
        public Symbol PrecedenceSymbol;
        public int ID;
        public bool CanReduce;
        public Rule NextLHSymbol;
        public Rule Next;
    }
}
