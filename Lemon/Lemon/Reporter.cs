using System;
using System.IO;
using System.Collections.Generic;

namespace Lemon
{
    public class Reporter
    {
        private static bool _showPrecedenceConflict = false;

        private static void ConfigPrint(TextWriter w, Config config)
        {
            var rule = config.Rule;
            w.Write("{0} ::=", rule.LHSymbol.Name);
            for (var i = 0; i <= rule.RHSymbols.Length; i++)
            {
                if (i == config.Dot)
                    w.Write(" *");
                if (i == rule.RHSymbols.Length)
                    break;
                var symbol = rule.RHSymbols[i];
                w.Write(" {0}", symbol.Name);
                if (symbol.Type == SymbolType.MultiTerminal)
                    for (var j = 1; j < symbol.Children.Length; j++)
                        w.Write("|{0}", symbol.Children[j].Name);
            }
        }

#if true
        /* Print a set */
        private static void SetPrint(TextWriter w, HashSet<int> set, Context ctx)
        {
            var spacer = string.Empty;
            w.Write("{0,12}[", string.Empty);
            for (var i = 0; i < ctx.Terminals; i++)
                if (set.Contains(i))
                {
                    w.Write("{0}{1}", spacer, ctx.Symbols[i].Name);
                    spacer = " ";
                }
            w.WriteLine("]");
        }

        /* Print a plink chain */
        private static void PlinkPrint(TextWriter w, List<Config> configs, string tag)
        {
            foreach (var config in configs)
            {
                w.Write("{0,12}{1} (state {2,2}) ", string.Empty, tag, config.State.ID);
                ConfigPrint(w, config);
                w.WriteLine();
            }
        }
#endif

        /* Print an action to the given file descriptor.  Return FALSE if nothing was actually printed. */
        private static bool PrintAction(Action action, TextWriter w, int indent)
        {
            var result = true;
            var firstParam = "{0," + indent.ToString() + "}";
            switch (action.Type)
            {
                case ActionType.Shift:
                    w.Write(firstParam + " shift  {1}", action.Symbol.Name, action.State.ID);
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
                    w.Write(firstParam + " reduce {1,-3} ** Parsing conflict **", action.Symbol.Name, action.Rule.ID);
                    break;
                case ActionType.SSConflict:
                    w.Write(firstParam + " shift  {1,-3} ** Parsing conflict **", action.Symbol.Name, action.State.ID);
                    break;
                case ActionType.SHResolved:
                    if (_showPrecedenceConflict)
                        w.Write(firstParam + " shift  {1,-3} -- dropped by precedence", action.Symbol.Name, action.State.ID);
                    else
                        result = false;
                    break;
                case ActionType.RDResolved:
                    if (_showPrecedenceConflict)
                        w.Write(firstParam + " reduce {1,-3} -- dropped by precedence", action.Symbol.Name, action.Rule.ID);
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
        public static void ReportOutput(Context ctx)
        {
            var filePath = Path.GetFullPath(ctx.Outname ?? ctx.Filename);
            filePath = Path.Combine(Path.GetDirectoryName(filePath), Path.GetFileNameWithoutExtension(filePath) + ".out");
            using (var w = new StreamWriter(filePath))
            {
                if (w == null)
                    return;
                for (var i = 0; i < ctx.States; i++)
                {
                    var states = ctx.Sorted[i];
                    w.WriteLine("State {0}:", states.ID);
                    var configs = (ctx.wantBasis ? states.Basises : states.Configs);
                    foreach (var config in configs)
                    {
                        if (config.Dot == config.Rule.RHSymbols.Length)
                            w.Write("    {0,5} ", "(" + config.Rule.ID + ")");
                        else
                            w.Write("          ");
                        ConfigPrint(w, config);
                        w.WriteLine();
#if false
                        SetPrint(w, config.FwSet, ctx);
                        PlinkPrint(w, config.Forwards, "To  ");
                        PlinkPrint(w, config.Basises, "From");
#endif
                    }
                    w.WriteLine();
                    foreach (var action in states.Actions)
                        if (PrintAction(action, w, 30))
                            w.WriteLine();
                    w.WriteLine();
                }
                w.WriteLine("----------------------------------------------------");
                w.WriteLine("Symbols:");
                for (var i = 0; i < ctx.Symbols.Length - 1; i++)
                {
                    var symbol = ctx.Symbols[i];
                    w.Write("  {0,3}: {1}", i, symbol.Name);
                    if (symbol.Type == SymbolType.NonTerminal)
                    {
                        w.Write(":");
                        if (symbol.Lambda)
                            w.Write(" <lambda>");
                        for (var j = 0; j < ctx.Terminals; j++)
                            if (symbol.FirstSet != null && symbol.FirstSet.Contains(j))
                                w.Write(" {0}", ctx.Symbols[j].Name);
                    }
                    w.WriteLine();
                }
            }
        }
    }
}
