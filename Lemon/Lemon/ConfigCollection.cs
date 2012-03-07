using System;
using System.Collections.Generic;

namespace Lemon
{
    public class ConfigCollection : Dictionary<Config, Config>
    {
        private List<Config> _items = new List<Config>();
        private List<Config> _basises = new List<Config>();
        private static readonly Config.KeyComparer _keyComparer = new Config.KeyComparer();

        public ConfigCollection()
            : base(new Config.KeyEqualityComparer()) { }

        public Config AddItem(Rule rule, int dot)
        {
            var key = new Config { Rule = rule, Dot = dot };
            Config config;
            if (!TryGetValue(key, out config))
            {
                config = key;
                config.FwSet = new HashSet<int>();
                config.Forwards = new List<Config>();
                config.Basises = new List<Config>();
                _items.Add(config);
                Add(key, config);
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
                config.FwSet = new HashSet<int>();
                config.Forwards = new List<Config>();
                config.Basises = new List<Config>();
                _items.Add(config);
                _basises.Add(config);
                Add(key, config);
            }
            return config;
        }

        public List<Config> SliceAllItems()
        {
            _items.Sort(_keyComparer);
            var lastItems = (_items.Count > 0 ? _items : null);
            _items = new List<Config>();
            return lastItems;
        }

        public List<Config> SliceAllBasises()
        {
            _basises.Sort(_keyComparer);
            var lastBasises = (_basises.Count > 0 ? _basises : null);
            _basises = new List<Config>();
            return lastBasises;
        }

        public void ListsReset()
        {
            _items.Clear();
            _basises.Clear();
            Clear();
        }

        public void Closure(Context ctx)
        {
            for (var itemIndex = 0; itemIndex < _items.Count; itemIndex++)
            {
                var item = _items[itemIndex];
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
                            item.Forwards.Add(newConfig);
                    }
                }
            }
        }

        internal void Eat(List<Config> configs)
        {
        }
    }
}
