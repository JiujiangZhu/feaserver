using System;
using System.Collections.Generic;

namespace Lemon
{
    public class Symbol : IComparer<Symbol>, IEquatable<Symbol>
    {
        public static SymbolCollection Symbols = new SymbolCollection();

        public string Name { get; internal set; }
        public int Index { get; private set; }
        public SymbolType Type { get; internal set; }
        public Rule Rule { get; set; }
        public Symbol Fallback { get; set; }
        public int Precedence { get; set; }
        public Association Association { get; set; }
        public HashSet<int> firstset { get; set; }
        public bool lambda { get; set; }
        public int useCnt { get; private set; }
        public string destructor { get; private set; }
        public int destLineno { get; private set; }
        public string datatype { get; private set; }
        public int dtnum { get; private set; }
        public Symbol[] subsym;

        static Symbol()
        {
            Symbol.New("$", false);
        }
        internal Symbol() { }

        public static Symbol New(string name) { return New(name, true); }
        public static Symbol New(string name, bool increaseUseCount)
        {
            Symbol sp;
            if (!Symbols.TryGetValue(name, out sp))
            {
                sp = new Symbol
                {
                    Name = name,
                    Type = (char.IsUpper(name[0]) ? SymbolType.Terminal : SymbolType.NonTerminal),
                    Precedence = -1,
                    Association = Association.Unknown,
                };
                Symbols.Add(name, sp);
            }
            if (increaseUseCount)
                sp.useCnt++;
            return sp;
        }

        public int Compare(Symbol x, Symbol y)
        {
            //const struct symbol **a = (const struct symbol **) _a;
            //const struct symbol **b = (const struct symbol **) _b;
            //int i1 = (**a).index + 10000000*((**a).name[0]>'Z');
            //int i2 = (**b).index + 10000000*((**b).name[0]>'Z');
            //assert( i1!=i2 || strcmp((**a).name,(**b).name)==0 );
            //return i1-i2;
            return 0;
        }

        public bool Equals(Symbol other)
        {
            if (base.Equals(other))
                return true;
            if ((Type != SymbolType.MultiTerminal) || (other.Type != SymbolType.MultiTerminal))
                return false;
            if (subsym.Length != other.subsym.Length)
                return false;
            for (var i = 0; i < subsym.Length; i++)
                if (subsym[i] != other.subsym[i])
                    return false;
            return true;
        }
    }
}
