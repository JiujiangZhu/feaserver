using System;
using System.IO;
using System.Collections.Generic;

namespace Lemon
{
    public class Reporter
    {
        private static bool _showPrecedenceConflict;

        private static void ConfigPrint(TextWriter w, Config config)
        {
            var rp = config.Rule;
            w.Write("{0} ::=", rp.LHSymbol.Name);
            for (var i = 0; i <= rp.nrhs; i++)
            {
                if (i == config.Dot)
                    w.Write(" *");
                if (i == rp.nrhs)
                    break;
                var sp = rp.RHSymbols[i];
                w.Write(" {0}", sp.Name);
                if (sp.Type == SymbolType.MultiTerminal)
                    for (var j = 1; j < sp.Children.Length; j++)
                        w.Write("|{0}", sp.Children[j].Name);
            }
        }

        /* #define TEST */
#if true
        /* Print a set */
        private static void SetPrint(TextWriter w, HashSet<int> set, Context ctx)
        {
            var spacer = string.Empty;
            w.Write("{12:0}[", string.Empty);
            for (var i = 0; i < ctx.Terminals; i++)
                if (set.Contains(i))
                {
                    w.Write("%s%s", spacer, ctx.Symbols[i].Name);
                    spacer = " ";
                }
            w.Write("]\n");
        }

        /* Print a plink chain */
        private static void PlinkPrint(TextWriter w, LinkedList<Config> configs, string tag)
        {
            foreach (var config in configs)
            {
                w.Write("{12:0}{1} (state {2:2}) ", string.Empty, tag, config.State.statenum);
                ConfigPrint(w, config);
                w.Write("\n");
            }
        }
#endif

        /* Print an action to the given file descriptor.  Return FALSE if nothing was actually printed. */
        private static bool PrintAction(Action action, TextWriter w, int indent)
        {
            var result = true;
            var firstParam = "{" + indent.ToString() + ":0}";
            switch (action.Type)
            {
                case ActionType.Shift:
                    w.Write(firstParam + " shift  {1}", action.Symbol.Name, action.State.statenum);
                    break;
                case ActionType.Reduce:
                    w.Write(firstParam + " reduce {1}", action.Symbol.Name, action.Rule.ID);
                    break;
                case ActionType.Accept:
                    w.Write(firstParam + " accept", action.Symbol.Name);
                    break;
                case ActionType.Error:
                    w.Write(firstParam + " error", action.Symbol.Name);
                    break;
                case ActionType.SRConflict:
                case ActionType.RRConflict:
                    w.Write(firstParam + " reduce {-3:1} ** Parsing conflict **", action.Symbol.Name, action.Rule.ID);
                    break;
                case ActionType.SSConflict:
                    w.Write(firstParam + " shift  {-3:1} ** Parsing conflict **", action.Symbol.Name, action.State.statenum);
                    break;
                case ActionType.SHResolved:
                    if (_showPrecedenceConflict)
                        w.Write(firstParam + " shift  {-3:1} -- dropped by precedence", action.Symbol.Name, action.State.statenum);
                    else
                        result = false;
                    break;
                case ActionType.RDResolved:
                    if (_showPrecedenceConflict)
                        w.Write(firstParam + " reduce {-3:1} -- dropped by precedence", action.Symbol.Name, action.Rule.ID);
                    else
                        result = false;
                    break;
                case ActionType.NotUsed:
                    result = false;
                    break;
            }
            return result;
        }

        /* Generate the "y.output" log file */
        private static void ReportOutput(Context ctx)
        {
            using (TextWriter fp = null) //new file_open(lemp,".out","wb");
            {
                if (fp == null)
                    return;
                for (var i = 0; i < ctx.States; i++)
                {
                    var stp = ctx.Sorted[i];
                    fp.Write("State %d:\n", stp.statenum);
                    var cfp = (ctx.wantBasis ? stp.bp : stp.cfp);
                    while (cfp != null)
                    {
                        var buf = new char[20];
                        if (cfp.Dot == cfp.Rule.nrhs)
                        {
                            //sprintf(buf,"(%d)",cfp.rp.index);
                            fp.Write("    %5s ", buf);
                        }
                        else
                            fp.Write("          ");
                        ConfigPrint(fp, cfp);
                        fp.Write("\n");
#if true
                        SetPrint(fp, cfp.fws, ctx);
                        PlinkPrint(fp, cfp.fplp, "To  ");
                        PlinkPrint(fp, cfp.bplp, "From");
#endif
                        cfp = (ctx.wantBasis ? cfp.Prev : cfp.Next);
                    }
                    fp.Write("\n");
                    for (var ap = stp.ap; ap != null; ap = ap.Next)
                        if (PrintAction(ap, fp, 30))
                            fp.Write("\n");
                    fp.Write("\n");
                }
                fp.Write("----------------------------------------------------\n");
                fp.Write("Symbols:\n");
                for (var i = 0; i < ctx.Symbols.Length; i++)
                {
                    var symbol = ctx.Symbols[i];
                    fp.Write("  %3d: %s", i, symbol.Name);
                    if (symbol.Type == SymbolType.NonTerminal)
                    {
                        fp.Write(":");
                        if (symbol.Lambda)
                            fp.Write(" <lambda>");
                        for (var j = 0; j < ctx.Terminals; j++)
                            if ((symbol.FirstSet != null) && symbol.FirstSet.Contains(j))
                                fp.Write(" %s", ctx.Symbols[j].Name);
                    }
                    fp.Write("\n");
                }
            }
        }
    }
}
