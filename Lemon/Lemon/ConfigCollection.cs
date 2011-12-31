using System;
using System.Collections.Generic;

namespace Lemon
{
    public class ConfigCollection : Dictionary<Config, Config>
    {
        private List<Config> _items;
        private List<Config> _basisItems;

        public Config AddItem(Rule rule, int dot)
        {
            Config config;
            var key = new Config { Rule = rule, Dot = dot };
            if (!TryGetValue(key, out config))
            {
                config = key;
                _items.Add(config);
                Add(config, config);
            }
            return config;
        }

        public Config AddBasisItem(Rule rule, int dot)
        {
            Config config;
            var key = new Config { Rule = rule, Dot = dot };
            if (!TryGetValue(key, out config))
            {
                config = key;
                _items.Add(config);
                _basisItems.Add(config);
                Add(config, config);
            }
            return config;
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

        public void Closure(Context ctx)
        {
            foreach (var item in _items)
            {
                var rule = item.Rule;
                var dot = item.Dot;
                if (dot >= rule.nrhs)
                    continue;
                var symbol = rule.RHSymbols[dot];
                if (symbol.Type == SymbolType.NonTerminal)
                {
                    if (symbol.Rule == null && symbol != ctx.ErrorSymbol)
                    {
                        Console.Write("{0}: Nonterminal \"{1}\" has no rules.", rule.Lineno, symbol.Name);
                        ctx.Errors++;
                    }
                    for (var newrp = symbol.Rule; newrp != null; newrp = newrp.NextLHSymbol)
                    {
                        var newcfp = AddItem(newrp, 0);
                        var i = dot + 1;
                        for (; i < rule.nrhs; i++)
                        {
                            var xsp = rule.RHSymbols[i];
                            if (xsp.Type == SymbolType.Terminal)
                            {
                                newcfp.fws.Add(xsp.ID);
                                break;
                            }
                            else if (xsp.Type == SymbolType.MultiTerminal)
                            {
                                foreach (var subSymbol in xsp.Children)
                                    newcfp.fws.Add(subSymbol.ID);
                                break;
                            }
                            else
                            {
                                newcfp.fws.Union(xsp.FirstSet);
                                if (!xsp.Lambda)
                                    break;
                            }
                        }
                        if (i == rule.nrhs)
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
