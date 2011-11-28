using System;
using System.Collections.Generic;

namespace Lemon
{
    public class ConfigCollection : Dictionary<Config, Config>
    {
        private List<Config> _items;
        private List<Config> _basisItems;

        public Config AddItem(Rule rp, int dot)
        {
            Config cf;
            var key = new Config { rp = rp, dot = dot };
            if (!TryGetValue(key, out cf))
            {
                cf = key;
                _items.Add(cf);
                Add(cf, cf);
            }
            return cf;
        }

        public Config AddBasisItem(Rule rp, int dot)
        {
            Config cf;
            var key = new Config { rp = rp, dot = dot };
            if (!TryGetValue(key, out cf))
            {
                cf = key;
                _items.Add(cf);
                _basisItems.Add(cf);
                Add(cf, cf);
            }
            return cf;
        }

        public Config GetItem()
        {
            _items.Sort();
            _items = new List<Config>();
            return (_items.Count > 0 ? _items[0] : null);
        }

        public Config GetBasisItem()
        {
            _basisItems.Sort();
            _basisItems = new List<Config>();
            return (_basisItems.Count > 0 ? _basisItems[0] : null);
        }

        public void ListsInit()
        {
            _items = new List<Config>();
            _basisItems = new List<Config>();
        }

        public void ListsReset()
        {
            _items = new List<Config>();
            _basisItems = new List<Config>();
            Clear();
        }

        public void Closure(Lemon lemon)
        {
            foreach (var item in _items)
            {
                var rp = item.rp;
                var dot = item.dot;
                if (dot >= rp.nrhs)
                    continue;
                var sp = rp.rhs[dot];
                if (sp.Type == SymbolType.NonTerminal)
                {
                    if ((sp.Rule == null) && (sp != lemon.errsym))
                    {
                        Console.Write("{0}: Nonterminal \"{1}\" has no rules.", rp.line, sp.Name);
                        lemon.errorcnt++;
                    }
                    for (var newrp = sp.Rule; newrp != null; newrp = newrp.nextlhs)
                    {
                        var newcfp = AddItem(newrp, 0);
                        var i = dot + 1;
                        for (; i < rp.nrhs; i++)
                        {
                            var xsp = rp.rhs[i];
                            if (xsp.Type == SymbolType.Terminal)
                            {
                                newcfp.fws.Add(xsp.Index);
                                break;
                            }
                            else if (xsp.Type == SymbolType.MultiTerminal)
                            {
                                foreach (var subSymbol in xsp.subsym)
                                    newcfp.fws.Add(subSymbol.Index);
                                break;
                            }
                            else
                            {
                                newcfp.fws.Union(xsp.firstset);
                                if (!xsp.lambda)
                                    break;
                            }
                        }
                        if (i == rp.nrhs)
                            item.fplp.AddFirst(newcfp);
                    }
                }
            }
        }

        internal void Eat(Config cfp)
        {
        }
    }
}
