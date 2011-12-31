using System;
using System.Linq;

namespace Lemon
{
    public class Program
    {
        public static void Main()
        {
            //          static int version = 0;
            var rpflag = true;
            //static int basisflag = 0;
            //static int compress = 0;
            //static int quiet = 0;
            var statistics = false;
            //static int mhflag = 0;
            //static int nolinenosflag = 0;
            //static int noResort = 0;
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

            var lem = new Context
            {
                argv0 = "argv0",
                Filename = @"..\..\..\parse.y",
                wantBasis = false,
                NoShowLinenos = false,
            };

            /* Parse the input file */
            Parser.Parse(lem);
            if (lem.Errors > 0) Environment.Exit(lem.Errors);
            if (lem.Rules == 0)
            {
                Console.WriteLine("Empty grammar.");
                Environment.Exit(1);
            }

            ///* Count and index the symbols of the grammar */
            //lem.nsymbol = Symbol_count();
            Symbol.New("{default}");
            lem.Symbols = Symbol.ToSymbolArray(out lem.Terminals);

            ///* Generate a reprint of the grammar, if requested on the command line */
            if (rpflag)
                lem.Reprint();
            else
            {
                //  /* Initialize the size for all follow and first sets */
                //  SetSize(lem.nterminal+1);

                //  /* Find the precedence for every production rule (that has one) */
                //  FindRulePrecedences(&lem);

                //  /* Compute the lambda-nonterminals and the first-sets for every
                //  ** nonterminal */
                //  FindFirstSets(&lem);

                //  /* Compute all LR(0) states.  Also record follow-set propagation
                //  ** links so that the follow-set can be computed later */
                //  lem.nstate = 0;
                //  FindStates(&lem);
                //  lem.sorted = State_arrayof();

                //  /* Tie up loose ends on the propagation links */
                //  FindLinks(&lem);

                //  /* Compute the follow set of every reducible configuration */
                //  FindFollowSets(&lem);

                //  /* Compute the action tables */
                //  FindActions(&lem);

                //  /* Compress the action tables */
                //  if( compress==0 )
                //      CompressTables(&lem);

                //  /* Reorder and renumber the states so that states with fewer choices occur at the end.  This is an optimization that helps make the generated parser tables smaller. */
                //  if( noResort==0 )
                //      ResortStates(&lem);

                //  /* Generate a report of the parser generated.  (the "y.output" file) */
                //  if( !quiet )
                //      ReportOutput(&lem);

                //  /* Generate the source code for the parser */
                //  ReportTable(&lem, mhflag);

                //  /* Produce a header file for use by the scanner.  (This step is omitted if the "-m" option is used because makeheaders will generate the file for us.) */
                //  if( !mhflag ) ReportHeader(&lem);
            }
            if (statistics)
            {
                Console.WriteLine("Parser statistics: {0} terminals, {1} nonterminals, {2} rules", lem.Terminals, lem.Symbols.Length - lem.Terminals, lem.Rules);
                Console.WriteLine("                   {0} states, {1} parser table entries, {2} conflicts", lem.States, lem.TableSize, lem.Conflicts);
            }
            if (lem.Conflicts > 0)
                Console.WriteLine("{0} parsing conflicts.", lem.Conflicts);
            /* return 0 on success, 1 on failure. */
            var exitcode = ((lem.Errors > 0) || (lem.Conflicts > 0) ? 1 : 0);
            Environment.Exit(exitcode);
        }
    }
}
