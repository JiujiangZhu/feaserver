using System;
using System.Collections.Generic;

namespace Lemon
{
    public class ConfigCollection : Dictionary<Config, Config>
    {
        private List<Config> _items = new List<Config>();
        private List<Config> _basis = new List<Config>();

        public ConfigCollection()
            : base(new Config.KeyComparer()) { }

        public Config AddItem(Rule rule, int dot)
        {
            var key = new Config { Rule = rule, Dot = dot };
            Config config;
            if (!TryGetValue(key, out config))
            {
                config = key;
                _items.Add(config);
                Add(config, config);
            }
            return config;
        }

        public Config AddBasis(Rule rule, int dot)
        {
            var key = new Config { Rule = rule, Dot = dot };
            Config config;
            if (!TryGetValue(key, out config))
            {
                config = key;
                _items.Add(config);
                _basis.Add(config);
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

        public Config GetBasis()
        {
            _basis.Sort();
            _basis = new List<Config>();
            return (_basis.Count > 0 ? _basis[0] : null);
        }

        public void ListsReset()
        {
            _items = new List<Config>();
            _basis = new List<Config>();
            Clear();
        }

        public void Closure(Context ctx)
        {
            foreach (var item in _items)
            {
                var rule = item.Rule;
                var dot = item.Dot;
                if (dot >= rule.RHSymbols.Length)
                    continue;
                var symbol = rule.RHSymbols[dot];
                if (symbol.Type == SymbolType.NonTerminal)
                {
                    if (symbol.Rule == null && symbol != ctx.ErrorSymbol)
                    {
                        Console.Write("{0}: Nonterminal \"{1}\" has no rules.", rule.Lineno, symbol.Name);
                        ctx.Errors++;
                    }
                    for (var newRule = symbol.Rule; newRule != null; newRule = newRule.NextLHSymbol)
                    {
                        var newConfig = AddItem(newRule, 0);
                        var i = dot + 1;
                        for (; i < rule.RHSymbols.Length; i++)
                        {
                            var symbol2 = rule.RHSymbols[i];
                            if (symbol2.Type == SymbolType.Terminal)
                            {
                                newConfig.FwSet.Add(symbol2.ID);
                                break;
                            }
                            else if (symbol2.Type == SymbolType.MultiTerminal)
                            {
                                foreach (var childSymbol in symbol2.Children)
                                    newConfig.FwSet.Add(childSymbol.ID);
                                break;
                            }
                            else
                            {
                                newConfig.FwSet.Union(symbol2.FirstSet);
                                if (!symbol2.Lambda)
                                    break;
                            }
                        }
                        if (i == rule.RHSymbols.Length)
                            item.Forwards.AddFirst(newConfig);
                    }
                }
            }
        }

        internal void Eat(Config cfp)
        {
        }
    }
}
