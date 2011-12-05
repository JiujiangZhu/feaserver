using System;
using System.IO;
using System.Collections.Generic;

namespace Lemon
{
    public class Reporter
    {
        private static bool _showPrecedenceConflict;

        private static void ConfigPrint(TextWriter fp, Config cfp)
        {
            var rp = cfp.rp;
            fp.Write("{0} ::=", rp.lhs.Name);
            for (var i = 0; i <= rp.nrhs; i++)
            {
                if (i == cfp.dot)
                    fp.Write(" *");
                if (i == rp.nrhs)
                    break;
                var sp = rp.rhs[i];
                fp.Write(" {0}", sp.Name);
                if (sp.Type == SymbolType.MultiTerminal)
                    for (var j = 1; j < sp.subsym.Length; j++)
                        fp.Write("|{0}", sp.subsym[j].Name);
            }
        }

        /* #define TEST */
#if true
        /* Print a set */
        private static void SetPrint(TextWriter fp, HashSet<int> set, Lemon lemp)
        {
            var spacer = string.Empty;
            fp.Write("{12:0}[", string.Empty);
            for (var i = 0; i < lemp.nterminal; i++)
                if (set.Contains(i))
                {
                    fp.Write("%s%s", spacer, lemp.symbols[i].Name);
                    spacer = " ";
                }
            fp.Write("]\n");
        }

        /* Print a plink chain */
        private static void PlinkPrint(TextWriter fp, LinkedList<Config> plp, string tag)
        {
            foreach (var cfp in plp)
            {
                fp.Write("{12:0}{1} (state {2:2}) ", string.Empty, tag, cfp.stp.statenum);
                ConfigPrint(fp, cfp);
                fp.Write("\n");
            }
        }
#endif

        /* Print an action to the given file descriptor.  Return FALSE if nothing was actually printed. */
        private static bool PrintAction(Action ap, TextWriter fp, int indent)
        {
            var result = true;
            var firstParam = "{" + indent.ToString() + ":0}";
            switch (ap.type)
            {
                case ActionType.Shift:
                    fp.Write(firstParam + " shift  {1}", ap.sp.Name, ap.stp.statenum);
                    break;
                case ActionType.Reduce:
                    fp.Write(firstParam + " reduce {1}", ap.sp.Name, ap.rp.index);
                    break;
                case ActionType.Accept:
                    fp.Write(firstParam + " accept", ap.sp.Name);
                    break;
                case ActionType.Error:
                    fp.Write(firstParam + " error", ap.sp.Name);
                    break;
                case ActionType.SRConflict:
                case ActionType.RRConflict:
                    fp.Write(firstParam + " reduce {-3:1} ** Parsing conflict **", ap.sp.Name, ap.rp.index);
                    break;
                case ActionType.SSConflict:
                    fp.Write(firstParam + " shift  {-3:1} ** Parsing conflict **", ap.sp.Name, ap.stp.statenum);
                    break;
                case ActionType.SHResolved:
                    if (_showPrecedenceConflict)
                        fp.Write(firstParam + " shift  {-3:1} -- dropped by precedence", ap.sp.Name, ap.stp.statenum);
                    else
                        result = false;
                    break;
                case ActionType.RDResolved:
                    if (_showPrecedenceConflict)
                        fp.Write(firstParam + " reduce {-3:1} -- dropped by precedence", ap.sp.Name, ap.rp.index);
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
        private static void ReportOutput(Lemon lemp)
        {
            using (TextWriter fp = null) //new file_open(lemp,".out","wb");
            {
                if (fp == null)
                    return;
                for (var i = 0; i < lemp.nstate; i++)
                {
                    var stp = lemp.sorted[i];
                    fp.Write("State %d:\n", stp.statenum);
                    var cfp = (lemp.basisflag ? stp.bp : stp.cfp);
                    while (cfp != null)
                    {
                        var buf = new char[20];
                        if (cfp.dot == cfp.rp.nrhs)
                        {
                            //sprintf(buf,"(%d)",cfp.rp.index);
                            fp.Write("    %5s ", buf);
                        }
                        else
                            fp.Write("          ");
                        ConfigPrint(fp, cfp);
                        fp.Write("\n");
#if true
                        SetPrint(fp, cfp.fws, lemp);
                        PlinkPrint(fp, cfp.fplp, "To  ");
                        PlinkPrint(fp, cfp.bplp, "From");
#endif
                        cfp = (lemp.basisflag ? cfp.bp : cfp.next);
                    }
                    fp.Write("\n");
                    for (var ap = stp.ap; ap != null; ap = ap.next)
                        if (PrintAction(ap, fp, 30))
                            fp.Write("\n");
                    fp.Write("\n");
                }
                fp.Write("----------------------------------------------------\n");
                fp.Write("Symbols:\n");
                for (var i = 0; i < lemp.nsymbol; i++)
                {
                    var sp = lemp.symbols[i];
                    fp.Write("  %3d: %s", i, sp.Name);
                    if (sp.Type == SymbolType.NonTerminal)
                    {
                        fp.Write(":");
                        if (sp.lambda)
                            fp.Write(" <lambda>");
                        for (var j = 0; j < lemp.nterminal; j++)
                            if ((sp.firstset != null) && sp.firstset.Contains(j))
                                fp.Write(" %s", lemp.symbols[j].Name);
                    }
                    fp.Write("\n");
                }
            }
        }
    }
}
