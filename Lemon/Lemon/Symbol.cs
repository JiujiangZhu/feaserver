using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class Symbol // : IEquatable<Symbol>
    {
        public static SymbolCollection Symbols = new SymbolCollection();
        private static readonly IComparer<Symbol> Compare = new SymbolComparer();

        private class SymbolComparer : IComparer<Symbol>
        {
            public int Compare(Symbol x, Symbol y)
            {
                var xi = x.ID + (char.IsUpper(x.Name[0]) ? 0 : 10000000);
                var yi = y.ID + (char.IsUpper(y.Name[0]) ? 0 : 10000000);
                Debug.Assert(xi != yi || x.Name == y.Name);
                return xi - yi;
            }
        }

        public string Name { get; internal set; }
        public int ID { get; set; }
        public SymbolType Type { get; internal set; }
        public Rule Rule { get; set; }
        public Symbol Fallback { get; set; }
        public int Precedence { get; set; }
        public Association Association { get; set; }
        public HashSet<int> FirstSet { get; set; }
        public bool Lambda { get; set; }
        public int Uses { get; private set; }
        public string Destructor { get; internal set; }
        public int DestructorLineno { get; set; }
        public string DataType { get; set; }
        public int DataTypeID { get; internal set; }
        public Symbol[] Children;

        static Symbol() { Symbol.New("$", false); }
        internal Symbol() { }

        public static Symbol New(string name) { return New(name, true); }
        public static Symbol New(string name, bool increaseUses)
        {
            Symbol symbol;
            if (!Symbols.TryGetValue(name, out symbol))
            {
                symbol = new Symbol
                {
                    Name = name,
                    Type = (char.IsUpper(name[0]) ? SymbolType.Terminal : SymbolType.NonTerminal),
                    Precedence = -1,
                    Association = Association.Unknown,
                };
                Symbols.Add(name, symbol);
            }
            if (increaseUses)
                symbol.Uses++;
            return symbol;
        }

        public bool Equals(Symbol other)
        {
            if (base.Equals(other))
                return true;
            if ((Type != SymbolType.MultiTerminal) || (other.Type != SymbolType.MultiTerminal))
                return false;
            if (Children.Length != other.Children.Length)
                return false;
            for (var index = 0; index < Children.Length; index++)
                if (Children[index] != other.Children[index])
                    return false;
            return true;
        }

        public static Symbol[] ToSymbolArray(out int terminals)
        {
            var symbols = Symbols.Values.ToArray();
            for (var index = 0; index < symbols.Length; index++) symbols[index].ID = index;
            Array.Sort(symbols, 1, symbols.Length - 1, Symbol.Compare);
            for (var index = 0; index < symbols.Length; index++) symbols[index].ID = index;
            terminals = 1;
            for (; char.IsUpper(symbols[terminals].Name[0]); terminals++) { }
            return symbols;
        }
    }
}
