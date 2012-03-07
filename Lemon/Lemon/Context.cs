using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class Context
    {
        public static readonly ConfigCollection Configs = new ConfigCollection();
        public static readonly StateCollection States_ = new StateCollection();
        private static readonly Action.KeyComparer _actionComparer = new Action.KeyComparer();

        public State[] Sorted;
        public Rule Rule;
        public int States;
        public int Rules;
        //public int _Symbols;
        public int Terminals;
        public Symbol[] Symbols;
        public int Errors;
        public Symbol ErrorSymbol = Symbol.New("error", false);
        public Symbol Wildcard;
        public string Name;
        public string ExtraArg;
        public string TokenType;
        public string DefaultDataType;
        public string StartSymbol;
        public string StackSize;
        public string Include;
        public string SyntaxError;
        public string StackOverflow;
        public string ParseFailure;
        public string ParseAccept;
        public string ExtraCode;
        public string TokenDestructor;
        public string DefaultDestructor;
        public string Filename;
        public string Outname;
        public string TokenPrefix;
        public int Conflicts;
        public int TableSize;
        public bool wantBasis;
        public bool HasFallback;
        public bool NoShowLinenos;
        public string argv0;

        public static void ErrorMsg(ref int errors, string filename, int lineno, string format, params object[] args)
        {
            var msg = string.Format(format, args);
            Console.WriteLine("{0}.{1}:{2}", filename, lineno, msg);
            errors++;
        }

        public void BuildRulesPrecedences()
        {
            for (var rule = Rule; rule != null; rule = rule.Next)
                if (rule.PrecedenceSymbol == null)
                    for (var index = 0; index < rule.RHSymbols.Length && rule.PrecedenceSymbol == null; index++)
                    {
                        var symbol = rule.RHSymbols[index];
                        if (symbol.Type == SymbolType.MultiTerminal)
                        {
                            for (var childenIndex = 0; childenIndex < symbol.Children.Length; childenIndex++)
                                if (symbol.Children[childenIndex].Precedence >= 0)
                                {
                                    rule.PrecedenceSymbol = symbol.Children[childenIndex];
                                    break;
                                }
                        }
                        else if (symbol.Precedence >= 0)
                            rule.PrecedenceSymbol = rule.RHSymbols[index];
                    }
        }

        public void BuildFirstSets()
        {
            //for (var i = 0; i < Symbols.Length; i++)
            //    Symbols[i].Lambda = false;
            for (var i = Terminals; i < Symbols.Length; i++)
                Symbols[i].FirstSet = new HashSet<int>();
            /* First compute all lambdas */
            bool progress = false;
            do
            {
                progress = false;
                for (var rule = Rule; rule != null; rule = rule.Next)
                {
                    if (rule.LHSymbol.Lambda)
                        continue;
                    var i = 0;
                    for (; i < rule.RHSymbols.Length; i++)
                    {
                        var symbol = rule.RHSymbols[i];
                        if (symbol.Type != SymbolType.Terminal || !symbol.Lambda)
                            break;
                    }
                    if (i == rule.RHSymbols.Length)
                    {
                        rule.LHSymbol.Lambda = true;
                        progress = true;
                    }
                }
            } while (progress);
            /* Now compute all first sets */
            do
            {
                progress = false;
                for (var rule = Rule; rule != null; rule = rule.Next)
                {
                    var lhSymbol = rule.LHSymbol;
                    for (var i = 0; i < rule.RHSymbols.Length; i++)
                    {
                        var rhSymbol = rule.RHSymbols[i];
                        if (rhSymbol.Type == SymbolType.Terminal)
                        {
                            progress |= lhSymbol.FirstSet.Add(rhSymbol.ID);
                            break;
                        }
                        else if (rhSymbol.Type == SymbolType.MultiTerminal)
                        {
                            for (var j = 0; j < rhSymbol.Children.Length; j++)
                                progress |= lhSymbol.FirstSet.Add(rhSymbol.Children[j].ID);
                            break;
                        }
                        else if (lhSymbol == rhSymbol)
                        {
                            if (!lhSymbol.Lambda)
                                break;
                        }
                        else
                        {
                            progress |= lhSymbol.FirstSet.Union(rhSymbol.FirstSet);
                            if (!rhSymbol.Lambda)
                                break;
                        }
                    }
                }
            } while (progress);
        }

        public void BuildStates()
        {
            States = 0;
            /* Find the start symbol */
            Symbol symbol;
            if (StartSymbol != null)
            {
                symbol = Symbol.Symbols[StartSymbol];
                if (symbol == null)
                {
                    ErrorMsg(ref Errors, Filename, 0, "The specified start symbol \"{0}\" is not in a nonterminal of the grammar.  \"{1}\" will be used as the start symbol instead.", StartSymbol, Rule.LHSymbol.Name);
                    symbol = Rule.LHSymbol;
                }
            }
            else
                symbol = Rule.LHSymbol;
            /* Make sure the start symbol doesn't occur on the right-hand side of any rule.  Report an error if it does.  (YACC would generate a new start symbol in this case.) */
            for (var rule = Rule; rule != null; rule = rule.Next)
                for (var i = 0; i < rule.RHSymbols.Length; i++)
                    if (rule.RHSymbols[i] == symbol)
                        /* FIX ME:  Deal with multiterminals */
                        ErrorMsg(ref Errors, Filename, 0, "The start symbol \"{0}\" occurs on the right-hand side of a rule. This will result in a parser which does not work properly.", symbol.Name);
            /* The basis configuration set for the first state is all rules which have the start symbol as their left-hand side */
            for (var rule = symbol.Rule; rule != null; rule = rule.NextLHSymbol)
            {
                rule.LHSymbolStart = 1;
                var newConfig = Context.Configs.AddBasis(rule, 0);
                newConfig.FwSet.Add(0);
            }
            /* Compute the first state.  All other states will be computed automatically during the computation of the first one. The returned pointer to the first state is not used. */
            Context.States_.GetState(this);
        }

        public void BuildLinks()
        {
            /* Add to every propagate link a pointer back to the state to which the link is attached. */
            for (var i = 0; i < States; i++)
            {
                var state = Sorted[i];
                foreach (var config in state.Configs)
                    config.State = state;
            }
            /* Convert all backlinks into forward links.  Only the forward links are used in the follow-set computation. */
            for (var i = 0; i < States; i++)
            {
                var state = Sorted[i];
                foreach (var config in state.Configs)
                    foreach (var other in config.Basises)
                        other.Forwards.Add(config);
            }
        }

        public void BuildFollowSets()
        {
            for (var i = 0; i < States; i++)
                foreach (var config in Sorted[i].Configs)
                    config.Complete = false;
            bool progress = false;
            do
            {
                progress = false;
                for (var i = 0; i < States; i++)
                    foreach (var config in Sorted[i].Configs)
                    {
                        if (config.Complete)
                            continue;
                        foreach (var other in config.Forwards)
                        {
                            var change = other.FwSet.Union(config.FwSet);
                            if (change)
                            {
                                other.Complete = false;
                                progress = true;
                            }
                        }
                        config.Complete = true;
                    }
            } while (progress);
        }

        public void BuildActions()
        {
            /* Add all of the reduce actions. A reduce action is added for each element of the followset of a configuration which has its dot at the extreme right. */
            for (var i = 0; i < States; i++)
            {
                var states = Sorted[i];
                foreach (var config in states.Configs)
                    if (config.Rule.RHSymbols.Length == config.Dot)
                        for (var j = 0; j < Terminals; j++)
                            if (config.FwSet.Contains(j))
                                /* Add a reduce action to the state "stp" which will reduce by the rule "cfp->rp" if the lookahead symbol is "lemp->symbols[j]" */
                                states.Actions.Add(new Action
                                {
                                    Type = ActionType.Reduce,
                                    Symbol = Symbols[j],
                                    Rule = config.Rule,
                                });
            }

            /* Add the accepting token */
            var symbol2 = (StartSymbol != null ? (Symbol.Symbols[StartSymbol] ?? Rule.LHSymbol) : Rule.LHSymbol);
            /* Add to the first state (which is always the starting state of the finite state machine) an action to ACCEPT if the lookahead is the start nonterminal.  */
            Sorted[0].Actions.Add(new Action
            {
                Type = ActionType.Accept,
                Symbol = symbol2,
            });

            /* Resolve conflicts */
            for (var i = 0; i < States; i++)
            {
                var symbol = Sorted[i];
                Debug.Assert(symbol.Actions.Count > 0);
                symbol.Actions.Sort(_actionComparer);
                for (var action = symbol.Action; action != null && action.Next != null; action = action.Next)
                    for (var nap = action.Next; nap != null && nap.Symbol == action.Symbol; nap = nap.Next)
                        /* The two actions "ap" and "nap" have the same lookahead. Figure out which one should be used */
                        Conflicts += Action.ResolveConflict(action, nap, ErrorSymbol);
            }

            /* Report an error for each rule that can never be reduced. */
            for (var rule = Rule; rule != null; rule = rule.Next)
                rule.CanReduce = false;
            for (var i = 0; i < States; i++)
                for (var action = Sorted[i].Action; action != null; action = action.Next)
                    if (action.Type == ActionType.Reduce)
                        action.Rule.CanReduce = true;
            for (var rule = Rule; rule != null; rule = rule.Next)
            {
                if (rule.CanReduce)
                    continue;
                ErrorMsg(ref Errors, Filename, rule.RuleLineno, "This rule can not be reduced.\n");
            }
        }

        public void BuildShifts(State state)
        {
            /* Each configuration becomes complete after it contibutes to a successor state.  Initially, all configurations are incomplete */
            foreach (var config2 in state.Configs)
                config2.Complete = false;
            /* Loop through all configurations of the state "stp" */
            var stateConfigs = state.Configs;
            for (var configIndex = 0; configIndex < stateConfigs.Count; configIndex++)
            {
                var config = stateConfigs[configIndex];
                if (config.Complete)
                    continue;
                if (config.Dot >= config.Rule.RHSymbols.Length)
                    continue;
                Context.Configs.ListsReset();
                var symbol = config.Rule.RHSymbols[config.Dot];
                /* For every configuration in the state "stp" which has the symbol "sp" following its dot, add the same configuration to the basis set under construction but with the dot shifted one symbol to the right. */
                for (var basisConfigIndex = configIndex; basisConfigIndex < stateConfigs.Count; basisConfigIndex++)
                {
                    var basisConfig = stateConfigs[basisConfigIndex];
                    if (basisConfig.Complete)
                        continue;
                    if (basisConfig.Dot >= basisConfig.Rule.RHSymbols.Length)
                        continue;
                    var scanSymbol = basisConfig.Rule.RHSymbols[basisConfig.Dot];
                    if (scanSymbol != symbol)
                        continue;
                    basisConfig.Complete = true;
                    var newConfig = Context.Configs.AddBasis(basisConfig.Rule, basisConfig.Dot + 1);
                    newConfig.Basises.Add(basisConfig);
                }
                /* Get a pointer to the state described by the basis configuration set constructed in the preceding loop */
                var newState = Context.States_.GetState(this);
                /* The state "newstp" is reached from the state "stp" by a shift action on the symbol "sp" */
                if (symbol.Type == SymbolType.MultiTerminal)
                    for (var i = 0; i < symbol.Children.Length; i++)
                        state.Actions.Add(new Action
                        {
                            Type = ActionType.Shift,
                            Symbol = symbol.Children[i],
                            State = newState,
                        });
                else
                    state.Actions.Add(new Action
                    {
                        Type = ActionType.Shift,
                        Symbol = symbol,
                        State = newState,
                    });
            }
        }

        public void CompressTables()
        {
            for (var i = 0; i < States; i++)
            {
                var state = Sorted[i];
                var bestN = 0;
                Rule bestRule = null;
                var usesWildcard = false;
                var stateActions = state.Actions;
                for(var actionIndex = 0; actionIndex < stateActions.Count; actionIndex++)
                {
                    var action = stateActions[actionIndex];
                    if (action.Type == ActionType.Shift && action.Symbol == Wildcard)
                        usesWildcard = true;
                    if (action.Type != ActionType.Reduce)
                        continue;
                    var rule = action.Rule;
                    if (rule.LHSymbolStart > 0)
                        continue;
                    if (rule == bestRule)
                        continue;
                    var n = 1;
                    for (var actionIndex2 = actionIndex + 1; actionIndex2 < stateActions.Count; actionIndex2++)
                    {
                        var action2 = stateActions[actionIndex2];
                        if (action2.Type != ActionType.Reduce)
                            continue;
                        var rp2 = action2.Rule;
                        if (rp2 == bestRule)
                            continue;
                        if (rp2 == rule)
                            n++;
                    }
                    if (n > bestN)
                    {
                        bestN = n;
                        bestRule = rule;
                    }
                }
                /* Do not make a default if the number of rules to default is not at least 1 or if the wildcard token is a possible lookahead. */
                if (bestN < 1 || usesWildcard)
                    continue;
                /* Combine matching REDUCE actions into a single default */
                var actionIndex3 = 0;
                Action action3 = null;
                for (; actionIndex3 < stateActions.Count; actionIndex3++)
                {
                    action3 = stateActions[actionIndex3];
                    if (action3.Type == ActionType.Reduce && action3.Rule == bestRule)
                        break;
                }
                Debug.Assert(action3 != null);
                action3.Symbol = Symbol.New("{default}");
                for (ap3 = ap3.Next; ap3 != null; ap3 = ap3.Next)
                    if (ap3.Type == ActionType.Reduce && ap3.Rule == bestRule)
                        ap3.Type = ActionType.NotUsed;
                state.Actions.Sort(_actionComparer);
            }
        }

        public void Reprint()
        {
            Console.Write("// Reprint of input file \"{0}\".\n// Symbols:\n", Filename);
            var maxlen = 10;
            for (var i = 0; i < Symbols.Length; i++)
            {
                var symbol = Symbols[i];
                var len = symbol.Name.Length;
                if (len > maxlen)
                    maxlen = len;
            }
            var columns = 76 / (maxlen + 5);
            if (columns < 1)
                columns = 1;
            var skip = (Symbols.Length + columns - 1) / columns;
            for (var i = 0; i < skip; i++)
            {
                Console.Write("//");
                for (var j = i; j < Symbols.Length; j += skip)
                {
                    var symbol = Symbols[j];
                    Debug.Assert(symbol.ID == j);
                    var symbolName = (symbol.Name.Length < maxlen ? symbol.Name : symbol.Name.Substring(0, maxlen));
                    Console.Write(" {0,3} {1,-" + maxlen.ToString() + "}", j, symbolName);
                }
                Console.Write("\n");
            }
            for (var rule = Rule; rule != null; rule = rule.Next)
            {
                Console.Write("{0}", rule.LHSymbol.Name);
                if (rule.LHSymbolAlias != null)
                    Console.Write("({0})", rule.LHSymbolAlias);
                Console.Write(" ::=");
                for (var i = 0; i < rule.RHSymbols.Length; i++)
                {
                    var symbol = rule.RHSymbols[i];
                    Console.Write(" {0}", symbol.Name);
                    if (symbol.Type == SymbolType.MultiTerminal)
                        for (var j = 1; j < symbol.Children.Length; j++)
                            Console.Write("|{0}", symbol.Children[j].Name);
                    //if (rule.RHSymbolsAlias[i] != null) Console.Write("({0})", rule.RHSymbolsAlias[i]);
                }
                Console.Write(".");
                if (rule.PrecedenceSymbol != null) Console.Write(" [{0}]", rule.PrecedenceSymbol.Name);
                //if (rule.Code != null) Console.Write("\n    {0}", rule.Code);
                Console.Write("\n");
            }
        }
    }
}
