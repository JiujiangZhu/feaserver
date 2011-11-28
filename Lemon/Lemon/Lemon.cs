using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class Lemon
    {
        public State[] sorted;
        public Rule rule;
        public int nstate;
        public int nrule;
        public int nsymbol;
        public int nterminal;
        public Symbol[] symbols;
        public int errorcnt;
        public Symbol errsym = Symbol.New("error", false);
        public Symbol wildcard;
        public string name;
        public string arg;
        public string tokentype;
        public string vartype;
        public string start;
        public string stacksize;
        public string include;
        public string error;
        public string overflow;
        public string failure;
        public string accept;
        public string extracode;
        public string tokendest;
        public string vardest;
        public string filename;
        public string outname;
        public string tokenprefix;
        public int nconflict;
        public int tablesize;
        public bool basisflag;
        public int has_fallback;
        public bool nolinenosflag;
        public string argv0;

        public static void ErrorMsg(ref int errorCount, string filename, int lineno, string format, params object[] args)
        {
            var msg = string.Format(format, args);
            Console.WriteLine("{0}.{1}:{2}", filename, lineno, msg);
            if (errorCount != null)
                errorCount++;
        }

        public void FindRulePrecedences()
        {
            for (var rp = rule; rp != null; rp = rp.next)
                if (rp.precsym == null)
                    for (var i = 0; (i < rp.nrhs) && (rp.precsym == null); i++)
                    {
                        var sp = rp.rhs[i];
                        if (sp.Type == SymbolType.MultiTerminal)
                        {
                            for (var j = 0; j < sp.subsym.Length; j++)
                                if (sp.subsym[j].Precedence >= 0)
                                {
                                    rp.precsym = sp.subsym[j];
                                    break;
                                }
                        }
                        else if (sp.Precedence >= 0)
                            rp.precsym = rp.rhs[i];
                    }
        }

        public void FindFirstSets()
        {
            for (var i = 0; i < nsymbol; i++)
                symbols[i].lambda = false;

            for (var i = nterminal; i < nsymbol; i++)
                symbols[i].firstset = new HashSet<int>();

            /* First compute all lambdas */
            bool progress = false;
            do
            {
                progress = false;
                for (var rp = rule; rp != null; rp = rp.next)
                {
                    if (rp.lhs.lambda)
                        continue;
                    var i = 0;
                    for (; i < rp.nrhs; i++)
                    {
                        var sp = rp.rhs[i];
                        if ((sp.Type != SymbolType.Terminal) || !sp.lambda)
                            break;
                    }
                    if (i == rp.nrhs)
                    {
                        rp.lhs.lambda = true;
                        progress = true;
                    }
                }
            } while (progress);

            /* Now compute all first sets */
            do
            {
                progress = false;
                for (var rp = rule; rp != null; rp = rp.next)
                {
                    var s1 = rp.lhs;
                    for (var i = 0; i < rp.nrhs; i++)
                    {
                        var s2 = rp.rhs[i];
                        if (s2.Type == SymbolType.Terminal)
                        {
                            progress |= s1.firstset.Add(s2.Index);
                            break;
                        }
                        else if (s2.Type == SymbolType.MultiTerminal)
                        {
                            for (var j = 0; j < s2.subsym.Length; j++)
                                progress |= s1.firstset.Add(s2.subsym[j].Index);
                            break;
                        }
                        else if (s1 == s2)
                        {
                            if (!s1.lambda)
                                break;
                        }
                        else
                        {
                            progress |= s1.firstset.Union(s2.firstset);
                            if (!s2.lambda)
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
            if (start != null)
            {
                sp = Symbol.Symbols[start];
                if (sp == null)
                {
                    ErrorMsg(ref errorcnt, filename, 0, "The specified start symbol \"{0}\" is not in a nonterminal of the grammar.  \"{1}\" will be used as the start symbol instead.", start, rule.lhs.Name);
                    sp = rule.lhs;
                }
            }
            else
                sp = rule.lhs;

            /* Make sure the start symbol doesn't occur on the right-hand side of any rule.  Report an error if it does.  (YACC would generate a new start symbol in this case.) */
            for (var rp = rule; rp != null; rp = rp.next)
                for (var i = 0; i < rp.nrhs; i++)
                    if (rp.rhs[i] == sp)
                        /* FIX ME:  Deal with multiterminals */
                        ErrorMsg(ref errorcnt, filename, 0, "The start symbol \"{0}\" occurs on the right-hand side of a rule. This will result in a parser which does not work properly.", sp.Name);

            /* The basis configuration set for the first state is all rules which have the start symbol as their left-hand side */
            for (var rp = sp.Rule; rp != null; rp = rp.nextlhs)
            {
                rp.lhsStart = 1;
                var newcfp = Config.Configs.AddBasisItem(rp, 0);
                newcfp.fws.Add(0);
            }

            /* Compute the first state.  All other states will be computed automatically during the computation of the first one. The returned pointer to the first state is not used. */
            State.States.GetState(this);
        }

        public void FindLinks()
        {
            /* Add to every propagate link a pointer back to the state to which the link is attached. */
            for (var i = 0; i < nstate; i++)
            {
                var stp = sorted[i];
                for (var cfp = stp.cfp; cfp != null; cfp = cfp.next)
                    cfp.stp = stp;
            }

            /* Convert all backlinks into forward links.  Only the forward links are used in the follow-set computation. */
            for (var i = 0; i < nstate; i++)
            {
                var stp = sorted[i];
                for (var cfp = stp.cfp; cfp != null; cfp = cfp.next)
                    foreach (var other in cfp.bplp)
                        other.fplp.AddFirst(cfp);
            }
        }

        void FindFollowSets()
        {
            for (var i = 0; i < nstate; i++)
                for (var cfp = sorted[i].cfp; cfp != null; cfp = cfp.next)
                    cfp.Complete = false;

            bool progress = false;
            do
            {
                progress = false;
                for (var i = 0; i < nstate; i++)
                    for (var cfp = sorted[i].cfp; cfp != null; cfp = cfp.next)
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
            for (var i = 0; i < nstate; i++)
            {
                var stp = sorted[i];
                for (var cfp = stp.cfp; cfp != null; cfp = cfp.next)
                    if (cfp.rp.nrhs == cfp.dot)
                        for (var j = 0; j < nterminal; j++)
                            if (cfp.fws.Contains(j))
                                /* Add a reduce action to the state "stp" which will reduce by the rule "cfp->rp" if the lookahead symbol is "lemp->symbols[j]" */
                                new Action(ref stp.ap)
                                {
                                    type = ActionType.Reduce,
                                    sp = symbols[j],
                                    rp = cfp.rp,
                                };
            }

            /* Add the accepting token */
            var sp = (start != null ? (Symbol.Symbols[start] ?? rule.lhs) : rule.lhs);
            /* Add to the first state (which is always the starting state of the finite state machine) an action to ACCEPT if the lookahead is the start nonterminal.  */
            new Action(ref sorted[0].ap)
            {
                type = ActionType.Accept,
                sp = sp,
            };

            /* Resolve conflicts */
            for (var i = 0; i < nstate; i++)
            {
                var stp = sorted[i];
                // Debug.Assert( stp.ap != null);
                stp.ap = Action.Sort(stp.ap);
                for (var ap = stp.ap; ap != null && ap.next != null; ap = ap.next)
                    for (var nap = ap.next; nap != null && nap.sp == ap.sp; nap = nap.next)
                        /* The two actions "ap" and "nap" have the same lookahead. Figure out which one should be used */
                        nconflict += Action.ResolveConflict(ap, nap, errsym);
            }

            /* Report an error for each rule that can never be reduced. */
            for (var rp = rule; rp != null; rp = rp.next)
                rp.canReduce = false;
            for (var i = 0; i < nstate; i++)
                for (var ap = sorted[i].ap; ap != null; ap = ap.next)
                    if (ap.type == ActionType.Reduce)
                        ap.rp.canReduce = true;
            for (var rp = rule; rp != null; rp = rp.next)
            {
                if (rp.canReduce)
                    continue;
                ErrorMsg(ref errorcnt, filename, rp.ruleline, "This rule can not be reduced.\n");
            }
        }

        public void BuildShifts(State stp)
        {
            /* Each configuration becomes complete after it contibutes to a successor state.  Initially, all configurations are incomplete */
            for (var cfp = stp.cfp; cfp != null; cfp = cfp.next)
                cfp.Complete = false;

            /* Loop through all configurations of the state "stp" */
            for (var cfp = stp.cfp; cfp != null; cfp = cfp.next)
            {
                if (cfp.Complete)
                    continue;
                if (cfp.dot >= cfp.rp.nrhs)
                    continue;
                Config.Configs.ListsReset();
                var sp = cfp.rp.rhs[cfp.dot];

                /* For every configuration in the state "stp" which has the symbol "sp" following its dot, add the same configuration to the basis set under construction but with the dot shifted one symbol to the right. */
                for (var bcfp = cfp; bcfp != null; bcfp = bcfp.next)
                {
                    if (bcfp.Complete)
                        continue;
                    if (bcfp.dot >= bcfp.rp.nrhs)
                        continue;
                    var bsp = bcfp.rp.rhs[bcfp.dot];
                    if (!bsp.Equals(sp))
                        continue;
                    bcfp.Complete = true;
                    var newcfg = Config.Configs.AddBasisItem(bcfp.rp, bcfp.dot + 1);
                    newcfg.bplp.AddFirst(bcfp);
                }

                /* Get a pointer to the state described by the basis configuration set constructed in the preceding loop */
                var newstp = State.States.GetState(this);

                /* The state "newstp" is reached from the state "stp" by a shift action on the symbol "sp" */
                if (sp.Type == SymbolType.MultiTerminal)
                    for (var i = 0; i < sp.subsym.Length; i++)
                        new Action(ref stp.ap)
                        {
                            type = ActionType.Shift,
                            sp = sp.subsym[i],
                            stp = newstp,
                        };
                else
                    new Action(ref stp.ap)
                    {
                        type = ActionType.Shift,
                        sp = sp,
                        stp = newstp,
                    };
            }
        }

        public void CompressTables()
        {
            for (var i = 0; i < nstate; i++)
            {
                var stp = sorted[i];
                var nbest = 0;
                Rule rbest = null;
                var usesWildcard = false;
                for (var ap = stp.ap; ap != null; ap = ap.next)
                {
                    if ((ap.type == ActionType.Shift) && (ap.sp == wildcard))
                        usesWildcard = true;
                    if (ap.type != ActionType.Reduce) 
                        continue;
                    var rp = ap.rp;
                    if (rp.lhsStart != null) 
                        continue;
                    if (rp == rbest) 
                        continue;
                    var n = 1;
                    for (var ap2 = ap.next; ap2 != null; ap2 = ap2.next)
                    {
                        if (ap2.type != ActionType.Reduce) 
                            continue;
                        var rp2 = ap2.rp;
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
                for (; ap3 != null; ap3 = ap3.next)
                    if ((ap3.type == ActionType.Reduce) && (ap3.rp == rbest))
                        break;
                Debug.Assert(ap3 != null);
                ap3.sp = Symbol.New("{default}");
                for (ap3 = ap3.next; ap3 != null; ap3 = ap3.next)
                    if ((ap3.type == ActionType.Reduce) && (ap3.rp == rbest))
                        ap3.type = ActionType.NotUsed;
                stp.ap = Action.Sort(stp.ap);
            }
        }

        public void Reprint()
        {
            Console.Write("// Reprint of input file \"{0}\".\n// Symbols:\n", filename);
            var maxlen = 10;
            for (var i = 0; i < nsymbol; i++)
            {
                var sp = symbols[i];
                var len = sp.Name.Length;
                if (len > maxlen)
                    maxlen = len;
            }
            var ncolumns = 76 / (maxlen + 5);
            if (ncolumns < 1)
                ncolumns = 1;
            var skip = (nsymbol + ncolumns - 1) / ncolumns;
            for (var i = 0; i < skip; i++)
            {
                Console.Write("//");
                for (var j = i; j < nsymbol; j += skip)
                {
                    var sp = symbols[j];
                    Debug.Assert(sp.Index == j);
                    Console.Write(" {0:3} %-*.*s", j, maxlen, maxlen, sp.Name);
                }
                Console.Write("\n");
            }
            for (var rp = rule; rp != null; rp = rp.next)
            {
                Console.Write("{0}", rp.lhs.Name);
                if (rp.lhsalias != null)
                    Console.Write("({0})", rp.lhsalias);
                Console.Write(" ::=");
                for (var i = 0; i < rp.nrhs; i++)
                {
                    var sp = rp.rhs[i];
                    Console.Write(" {0}", sp.Name);
                    if (sp.Type == SymbolType.MultiTerminal)
                        for (var j = 1; j < sp.subsym.Length; j++)
                            Console.Write("|{0}", sp.subsym[j].Name);
                    if (rp.rhsalias[i] != null)
                        Console.Write("(%s)", rp.rhsalias[i]);
                }
                Console.Write(".");
                if (rp.precsym != null)
                    Console.Write(" [%s]", rp.precsym.Name);
                if (rp.code != null)
                    Console.Write("\n    {0}", rp.code);
                Console.Write("\n");
            }
        }
    }
}
