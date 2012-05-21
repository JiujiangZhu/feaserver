using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class Symbol
    {
        private static readonly IComparer<Symbol> _keyComparer = new KeyComparer();

        private class KeyComparer : IComparer<Symbol>
        {
            public int Compare(Symbol x, Symbol y)
            {
                if (object.ReferenceEquals(x, y))
                    return 0;
                var xi = x.ID + (x.Name[0] > 'Z' ? 10000000 : 0);
                var yi = y.ID + (y.Name[0] > 'Z' ? 10000000 : 0);
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

        internal Symbol() { }

        public static Symbol New(string name) { return New(name, true); }
        public static Symbol New(string name, bool increaseUses)
        {
            Symbol symbol;
            if (!Context.AllSymbols.TryGetValue(name, out symbol))
            {
                symbol = new Symbol
                {
                    Name = name,
                    Type = (char.IsUpper(name[0]) ? SymbolType.Terminal : SymbolType.NonTerminal),
                    Precedence = -1,
                    Association = Association.Unknown,
                };
                Context.AllSymbols.Add(name, symbol);
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
            Symbol.New("{default}");
            var symbols = Context.AllSymbols.Values.ToArray();
            for (var index = 0; index < symbols.Length; index++) symbols[index].ID = index;
            Array.Sort(symbols, _keyComparer);
            for (var index = 0; index < symbols.Length; index++) symbols[index].ID = index;
            var i = 1;
            for (; char.IsUpper(symbols[i].Name[0]); i++) { }
            terminals = i;
            return symbols;
        }
    }
}
