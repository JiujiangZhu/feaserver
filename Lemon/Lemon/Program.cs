using System;

namespace Lemon
{
    public class Program
    {
        public static void Main()
        {
            //          static int version = 0;
            //static int rpflag = 0;
            //static int basisflag = 0;
            //static int compress = 0;
            //static int quiet = 0;
            //static int statistics = 0;
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


            var lemon = new Lemon
            {
                argv0 = "argv0",
                filename = "optArg0",
                basisflag = false,
                nolinenosflag = false,
            };

            ///* Parse the input file */
            //Parse(&lem);
            //if( lem.errorcnt )
            //    exit(lem.errorcnt);
            //if( lem.nrule==0 ){
            //  fprintf(stderr,"Empty grammar.\n");
            //  exit(1);
            //}

            ///* Count and index the symbols of the grammar */
            //lem.nsymbol = Symbol_count();
            //Symbol_new("{default}");
            //lem.symbols = Symbol_arrayof();
            //for(i=0; i<=lem.nsymbol; i++) lem.symbols[i]->index = i;
            //qsort(lem.symbols,lem.nsymbol+1,sizeof(struct symbol*), Symbolcmpp);
            //for(i=0; i<=lem.nsymbol; i++) lem.symbols[i]->index = i;
            //for(i=1; isupper(lem.symbols[i]->name[0]); i++);
            //lem.nterminal = i;

            ///* Generate a reprint of the grammar, if requested on the command line */
            //if( rpflag ){
            //  Reprint(&lem);
            //}else{
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
            //}
            //if( statistics ){
            //  printf("Parser statistics: %d terminals, %d nonterminals, %d rules\n", lem.nterminal, lem.nsymbol - lem.nterminal, lem.nrule);
            //  printf("                   %d states, %d parser table entries, %d conflicts\n", lem.nstate, lem.tablesize, lem.nconflict);
            //}
            //if( lem.nconflict > 0 ){
            //  fprintf(stderr,"%d parsing conflicts.\n",lem.nconflict);
            //}

            ///* return 0 on success, 1 on failure. */
            //exitcode = ((lem.errorcnt > 0) || (lem.nconflict > 0)) ? 1 : 0;
            //exit(exitcode);
            //return (exitcode);
        }
    }
}
