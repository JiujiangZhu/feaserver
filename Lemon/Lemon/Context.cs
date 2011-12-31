using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class Context
    {
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

        public void EnsureRulesPrecedences()
        {
            for (var rule = Rule; rule != null; rule = rule.Next)
                if (rule.PrecedenceSymbol == null)
                    for (var index = 0; index < rule.nrhs && rule.PrecedenceSymbol == null; index++)
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

        public void FindFirstSets()
        {
            for (var i = 0; i < Symbols.Length; i++)
                Symbols[i].Lambda = false;

            for (var i = Terminals; i < Symbols.Length; i++)
                Symbols[i].FirstSet = new HashSet<int>();

            /* First compute all lambdas */
            bool progress = false;
            do
            {
                progress = false;
                for (var rp = Rule; rp != null; rp = rp.Next)
                {
                    if (rp.LHSymbol.Lambda)
                        continue;
                    var i = 0;
                    for (; i < rp.nrhs; i++)
                    {
                        var sp = rp.RHSymbols[i];
                        if ((sp.Type != SymbolType.Terminal) || !sp.Lambda)
                            break;
                    }
                    if (i == rp.nrhs)
                    {
                        rp.LHSymbol.Lambda = true;
                        progress = true;
                    }
                }
            } while (progress);

            /* Now compute all first sets */
            do
            {
                progress = false;
                for (var rp = Rule; rp != null; rp = rp.Next)
                {
                    var s1 = rp.LHSymbol;
                    for (var i = 0; i < rp.nrhs; i++)
                    {
                        var s2 = rp.RHSymbols[i];
                        if (s2.Type == SymbolType.Terminal)
                        {
                            progress |= s1.FirstSet.Add(s2.ID);
                            break;
                        }
                        else if (s2.Type == SymbolType.MultiTerminal)
                        {
                            for (var j = 0; j < s2.Children.Length; j++)
                                progress |= s1.FirstSet.Add(s2.Children[j].ID);
                            break;
                        }
                        else if (s1 == s2)
                        {
                            if (!s1.Lambda)
                                break;
                        }
                        else
                        {
                            progress |= s1.FirstSet.Union(s2.FirstSet);
                            if (!s2.Lambda)
                                break;
                        }
                    }
                }
            } while (progress);
        }


        public void FindStates()
        {
            Config.Configs.ListsInit();

            /* Find the start symbol */
            Symbol sp;
            if (StartSymbol != null)
            {
                sp = Symbol.Symbols[StartSymbol];
                if (sp == null)
                {
                    ErrorMsg(ref Errors, Filename, 0, "The specified start symbol \"{0}\" is not in a nonterminal of the grammar.  \"{1}\" will be used as the start symbol instead.", StartSymbol, Rule.LHSymbol.Name);
                    sp = Rule.LHSymbol;
                }
            }
            else
                sp = Rule.LHSymbol;

            /* Make sure the start symbol doesn't occur on the right-hand side of any rule.  Report an error if it does.  (YACC would generate a new start symbol in this case.) */
            for (var rp = Rule; rp != null; rp = rp.Next)
                for (var i = 0; i < rp.nrhs; i++)
                    if (rp.RHSymbols[i] == sp)
                        /* FIX ME:  Deal with multiterminals */
                        ErrorMsg(ref Errors, Filename, 0, "The start symbol \"{0}\" occurs on the right-hand side of a rule. This will result in a parser which does not work properly.", sp.Name);

            /* The basis configuration set for the first state is all rules which have the start symbol as their left-hand side */
            for (var rp = sp.Rule; rp != null; rp = rp.NextLHSymbol)
            {
                rp.LhSymbolStart = 1;
                var newcfp = Config.Configs.AddBasisItem(rp, 0);
                newcfp.fws.Add(0);
            }

            /* Compute the first state.  All other states will be computed automatically during the computation of the first one. The returned pointer to the first state is not used. */
            State.States.GetState(this);
        }

        public void FindLinks()
        {
            /* Add to every propagate link a pointer back to the state to which the link is attached. */
            for (var i = 0; i < States; i++)
            {
                var stp = Sorted[i];
                for (var cfp = stp.cfp; cfp != null; cfp = cfp.Next)
                    cfp.State = stp;
            }

            /* Convert all backlinks into forward links.  Only the forward links are used in the follow-set computation. */
            for (var i = 0; i < States; i++)
            {
                var stp = Sorted[i];
                for (var cfp = stp.cfp; cfp != null; cfp = cfp.Next)
                    foreach (var other in cfp.bplp)
                        other.fplp.AddFirst(cfp);
            }
        }

        void FindFollowSets()
        {
            for (var i = 0; i < States; i++)
                for (var cfp = Sorted[i].cfp; cfp != null; cfp = cfp.Next)
                    cfp.Complete = false;

            bool progress = false;
            do
            {
                progress = false;
                for (var i = 0; i < States; i++)
                    for (var cfp = Sorted[i].cfp; cfp != null; cfp = cfp.Next)
                    {
                        if (cfp.Complete)
                            continue;
                        foreach (var other in cfp.fplp)
                        {
                            var change = other.fws.Union(cfp.fws);
                            if (change)
                            {
                                other.Complete = false;
                                progress = true;
                            }
                        }
                        cfp.Complete = true;
                    }
            } while (progress);
        }

        public void FindActions()
        {
            /* Add all of the reduce actions. A reduce action is added for each element of the followset of a configuration which has its dot at the extreme right. */
            for (var i = 0; i < States; i++)
            {
                var stp = Sorted[i];
                for (var cfp = stp.cfp; cfp != null; cfp = cfp.Next)
                    if (cfp.Rule.nrhs == cfp.Dot)
                        for (var j = 0; j < Terminals; j++)
                            if (cfp.fws.Contains(j))
                                /* Add a reduce action to the state "stp" which will reduce by the rule "cfp->rp" if the lookahead symbol is "lemp->symbols[j]" */
                                new Action(ref stp.ap)
                                {
                                    Type = ActionType.Reduce,
                                    Symbol = Symbols[j],
                                    Rule = cfp.Rule,
                                };
            }

            /* Add the accepting token */
            var sp = (StartSymbol != null ? (Symbol.Symbols[StartSymbol] ?? Rule.LHSymbol) : Rule.LHSymbol);
            /* Add to the first state (which is always the starting state of the finite state machine) an action to ACCEPT if the lookahead is the start nonterminal.  */
            new Action(ref Sorted[0].ap)
            {
                Type = ActionType.Accept,
                Symbol = sp,
            };

            /* Resolve conflicts */
            for (var i = 0; i < States; i++)
            {
                var stp = Sorted[i];
                // Debug.Assert( stp.ap != null);
                stp.ap = Action.Sort(stp.ap);
                for (var ap = stp.ap; ap != null && ap.Next != null; ap = ap.Next)
                    for (var nap = ap.Next; nap != null && nap.Symbol == ap.Symbol; nap = nap.Next)
                        /* The two actions "ap" and "nap" have the same lookahead. Figure out which one should be used */
                        Conflicts += Action.ResolveConflict(ap, nap, ErrorSymbol);
            }

            /* Report an error for each rule that can never be reduced. */
            for (var rp = Rule; rp != null; rp = rp.Next)
                rp.CanReduce = false;
            for (var i = 0; i < States; i++)
                for (var ap = Sorted[i].ap; ap != null; ap = ap.Next)
                    if (ap.Type == ActionType.Reduce)
                        ap.Rule.CanReduce = true;
            for (var rp = Rule; rp != null; rp = rp.Next)
            {
                if (rp.CanReduce)
                    continue;
                ErrorMsg(ref Errors, Filename, rp.RuleLineno, "This rule can not be reduced.\n");
            }
        }

        public void BuildShifts(State stp)
        {
            /* Each configuration becomes complete after it contibutes to a successor state.  Initially, all configurations are incomplete */
            for (var cfp = stp.cfp; cfp != null; cfp = cfp.Next)
                cfp.Complete = false;

            /* Loop through all configurations of the state "stp" */
            for (var cfp = stp.cfp; cfp != null; cfp = cfp.Next)
            {
                if (cfp.Complete)
                    continue;
                if (cfp.Dot >= cfp.Rule.nrhs)
                    continue;
                Config.Configs.ListsReset();
                var sp = cfp.Rule.RHSymbols[cfp.Dot];

                /* For every configuration in the state "stp" which has the symbol "sp" following its dot, add the same configuration to the basis set under construction but with the dot shifted one symbol to the right. */
                for (var bcfp = cfp; bcfp != null; bcfp = bcfp.Next)
                {
                    if (bcfp.Complete)
                        continue;
                    if (bcfp.Dot >= bcfp.Rule.nrhs)
                        continue;
                    var bsp = bcfp.Rule.RHSymbols[bcfp.Dot];
                    if (!bsp.Equals(sp))
                        continue;
                    bcfp.Complete = true;
                    var newcfg = Config.Configs.AddBasisItem(bcfp.Rule, bcfp.Dot + 1);
                    newcfg.bplp.AddFirst(bcfp);
                }

                /* Get a pointer to the state described by the basis configuration set constructed in the preceding loop */
                var newstp = State.States.GetState(this);

                /* The state "newstp" is reached from the state "stp" by a shift action on the symbol "sp" */
                if (sp.Type == SymbolType.MultiTerminal)
                    for (var i = 0; i < sp.Children.Length; i++)
                        new Action(ref stp.ap)
                        {
                            Type = ActionType.Shift,
                            Symbol = sp.Children[i],
                            State = newstp,
                        };
                else
                    new Action(ref stp.ap)
                    {
                        Type = ActionType.Shift,
                        Symbol = sp,
                        State = newstp,
                    };
            }
        }

        public void CompressTables()
        {
            for (var i = 0; i < States; i++)
            {
                var stp = Sorted[i];
                var nbest = 0;
                Rule rbest = null;
                var usesWildcard = false;
                for (var ap = stp.ap; ap != null; ap = ap.Next)
                {
                    if ((ap.Type == ActionType.Shift) && (ap.Symbol == Wildcard))
                        usesWildcard = true;
                    if (ap.Type != ActionType.Reduce)
                        continue;
                    var rp = ap.Rule;
                    if (rp.LhSymbolStart > 0)
                        continue;
                    if (rp == rbest)
                        continue;
                    var n = 1;
                    for (var ap2 = ap.Next; ap2 != null; ap2 = ap2.Next)
                    {
                        if (ap2.Type != ActionType.Reduce)
                            continue;
                        var rp2 = ap2.Rule;
                        if (rp2 == rbest)
                            continue;
                        if (rp2 == rp)
                            n++;
                    }
                    if (n > nbest)
                    {
                        nbest = n;
                        rbest = rp;
                    }
                }

                /* Do not make a default if the number of rules to default is not at least 1 or if the wildcard token is a possible lookahead. */
                if ((nbest < 1) || (usesWildcard))
                    continue;

                /* Combine matching REDUCE actions into a single default */
                var ap3 = stp.ap;
                for (; ap3 != null; ap3 = ap3.Next)
                    if ((ap3.Type == ActionType.Reduce) && (ap3.Rule == rbest))
                        break;
                Debug.Assert(ap3 != null);
                ap3.Symbol = Symbol.New("{default}");
                for (ap3 = ap3.Next; ap3 != null; ap3 = ap3.Next)
                    if ((ap3.Type == ActionType.Reduce) && (ap3.Rule == rbest))
                        ap3.Type = ActionType.NotUsed;
                stp.ap = Action.Sort(stp.ap);
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
                for (var i = 0; i < rule.nrhs; i++)
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
