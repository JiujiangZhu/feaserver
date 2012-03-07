using System;
using System.Linq;

namespace Lemon
{
    public class Program
    {
        public static void Main()
        {
            //int version = 0;
            var rpflag = false;
            //static int basisflag = 0;
            var compress = false;
            var quiet = false;
            var statistics = false;
            var mhflag = false;
            //var nolinenosflag = false;
            var noResort = false;
            //static struct s_options options[] = {
            //  {OPT_FLAG, "b", (char*)&basisflag, "Print only the basis in report."},
            //  {OPT_FLAG, "c", (char*)&compress, "Don't compress the action table."},
            //  {OPT_FSTR, "D", (char*)handle_D_option, "Define an %ifdef macro."},
            //  {OPT_FSTR, "T", (char*)handle_T_option, "Specify a template file."},
            //  {OPT_FLAG, "g", (char*)&rpflag, "Print grammar without actions."},
            //  {OPT_FLAG, "m", (char*)&mhflag, "Output a makeheaders compatible file."},
            //  {OPT_FLAG, "l", (char*)&nolinenosflag, "Do not print #line statements."},
            //  {OPT_FLAG, "p", (char*)&showPrecedenceConflict,
            //                  "Show conflicts resolved by precedence rules"},
            //  {OPT_FLAG, "q", (char*)&quiet, "(Quiet) Don't print the report file."},
            //  {OPT_FLAG, "r", (char*)&noResort, "Do not sort or renumber states"},
            //  {OPT_FLAG, "s", (char*)&statistics,
            //                                 "Print parser stats to standard output."},
            //  {OPT_FLAG,0,0,0}
            //};

            var ctx = new Context
            {
                argv0 = "argv0",
                Filename = @"..\..\..\parse.y",
                wantBasis = false,
                NoShowLinenos = false,
            };

            /* Parse the input file */
            Parser.Parse(ctx);
            if (ctx.Errors > 0) Environment.Exit(ctx.Errors);
            if (ctx.Rules == 0)
            {
                Console.WriteLine("Empty grammar.");
                Environment.Exit(1);
            }

            ///* Count and index the symbols of the grammar */
            //lem.nsymbol = Symbol_count();
            Symbol.New("{default}");
            ctx.Symbols = Symbol.ToSymbolArray(out ctx.Terminals);

            ///* Generate a reprint of the grammar, if requested on the command line */
            if (rpflag)
                ctx.Reprint();
            else
            {
                  ///* Initialize the size for all follow and first sets */
                  ////SetSize(lemon.nterminal+1);
                  /* Find the precedence for every production rule (that has one) */
                  ctx.BuildRulesPrecedences();
                  /* Compute the lambda-nonterminals and the first-sets for every nonterminal */
                  ctx.BuildFirstSets();
                  /* Compute all LR(0) states.  Also record follow-set propagation links so that the follow-set can be computed later */
                  ctx.BuildStates();
                  ////lemon.sorted = State_arrayof();
                  ///* Tie up loose ends on the propagation links */
                  //ctx.BuildLinks();
                  ///* Compute the follow set of every reducible configuration */
                  //ctx.BuildFollowSets();
                  ///* Compute the action tables */
                  //ctx.BuildActions();
                  ///* Compress the action tables */
                  //if (!compress)
                  //    ctx.CompressTables();
                  ///* Reorder and renumber the states so that states with fewer choices occur at the end.  This is an optimization that helps make the generated parser tables smaller. */
                  //if (!noResort)
                  //    ctx.ResortStates();
                  ///* Generate a report of the parser generated.  (the "y.output" file) */
                  //if (!quiet)
                  //    Reporter.ReportOutput(ctx);
                  ///* Generate the source code for the parser */
                  //ctx.ReportTable(mhflag);
                  ///* Produce a header file for use by the scanner.  (This step is omitted if the "-m" option is used because makeheaders will generate the file for us.) */
                  //if (!mhflag) Reporter.ReportHeader(ctx);
            }
            if (statistics)
            {
                Console.WriteLine("Parser statistics: {0} terminals, {1} nonterminals, {2} rules", ctx.Terminals, ctx.Symbols.Length - ctx.Terminals, ctx.Rules);
                Console.WriteLine("                   {0} states, {1} parser table entries, {2} conflicts", ctx.States, ctx.TableSize, ctx.Conflicts);
            }
            if (ctx.Conflicts > 0)
                Console.WriteLine("{0} parsing conflicts.", ctx.Conflicts);
            /* return 0 on success, 1 on failure. */
            var exitcode = ((ctx.Errors > 0) || (ctx.Conflicts > 0) ? 1 : 0);
            Environment.Exit(exitcode);
        }
    }
}
