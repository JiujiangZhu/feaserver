using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Lemon
{
    public partial class Emitter
    {
        public static readonly AX.KeyComparer _keyComparer = new AX.KeyComparer();

        /* The following routine emits code for the destructor for the symbol sp */
        private static void emit_destructor_code(StreamWriter w, Symbol symbol, Context ctx, ref int lineno)
        {
            string z;
            if (symbol.Type == SymbolType.Terminal)
            {
                z = ctx.TokenDestructor;
                if (z == null) return;
                w.WriteLine(ref lineno, "{");
            }
            else if (symbol.Destructor != null)
            {
                z = symbol.Destructor;
                w.WriteLine(ref lineno, "{");
                if (!ctx.NoShowLinenos) { lineno++; Template.tplt_linedir(w, symbol.DestructorLineno, ctx.Filename); }
            }
            else if (ctx.DefaultDestructor != null)
            {
                z = ctx.DefaultDestructor;
                if (z == null) return;
                w.WriteLine(ref lineno, "{");
            }
            else
                throw new InvalidOperationException();
            var cp = 0;
            for (; cp < z.Length; cp++)
            {
                if (z[cp] == '$' && z[cp + 1] == '$')
                {
                    w.Write("(yypminor.yy{0})", symbol.DataTypeID);
                    cp++;
                    continue;
                }
                if (z[cp] == '\n') lineno++;
                w.Write(z[cp]);
            }
            w.WriteLine(ref lineno);
            if (!ctx.NoShowLinenos) { lineno++; Template.tplt_linedir(w, lineno, ctx.Outname); }
            w.WriteLine(ref lineno, "}");
            return;
        }

        /* Return TRUE (non-zero) if the given symbol has a destructor. */
        private static bool has_destructor(Symbol symbol, Context ctx)
        {
            return (symbol.Type == SymbolType.Terminal ? ctx.TokenDestructor != null : ctx.DefaultDestructor != null || symbol.Destructor != null);
        }

        /* zCode is a string that is the action associated with a rule.  Expand the symbols in this string so that the refer to elements of the parser stack. */
        private static void translate_code(Context ctx, Rule rule)
        {
            var used = new bool[rule.RHSymbols.Length]; /* True for each RHS element which is used */
            var lhsused = false; /* True if the LHS element has been used */
            if (rule.Code == null) { rule.Code = "\n"; rule.Lineno = rule.RuleLineno; }
            var b = new StringBuilder();
            var z = rule.Code;
            for (var cp = 0; cp < z.Length; cp++)
            {
                if (char.IsLower(z[cp]) && (cp == 0 || (!char.IsLetterOrDigit(z[cp - 1]) && z[cp - 1] != '_')))
                {
                    var xp = cp + 1;
                    for (; char.IsLetterOrDigit(z[xp]) || z[xp] == '_'; xp++) ;
                    var saved = z[xp];
                    var xpLength = xp - cp;
                    if (rule.LHSymbolAlias != null && z.Substring(cp, xpLength) == rule.LHSymbolAlias)
                    {
                        b.AppendFormat("yygotominor.yy{0}", rule.LHSymbol.DataTypeID);
                        cp = xp;
                        lhsused = true;
                    }
                    else
                    {
                        for (var i = 0; i < rule.RHSymbols.Length; i++)
                            if (rule.RHSymbolsAlias[i] != null && z.Substring(cp, xpLength) == rule.RHSymbolsAlias[i])
                            {
                                /* If the argument is of the form @X then substituted the token number of X, not the value of X */
                                if (cp > 0 && z[cp - 1] == '@')
                                {
                                    b.Length--;
                                    b.AppendFormat("yymsp[{0}].major", i - rule.RHSymbols.Length + 1);
                                }
                                else
                                {
                                    var sp = rule.RHSymbols[i];
                                    var dtnum = (sp.Type == SymbolType.MultiTerminal ? sp.Children[0].DataTypeID : sp.DataTypeID);
                                    b.AppendFormat("yymsp[{0}].minor.yy{1}", i - rule.RHSymbols.Length + 1, dtnum);
                                }
                                cp = xp;
                                used[i] = true;
                                break;
                            }
                    }
                }
                b.Append(z.Substring(cp));
            }
            /* Check to make sure the LHS has been used */
            if (rule.LHSymbolAlias != null && !lhsused)
                Context.ErrorMsg(ref ctx.Errors, ctx.Filename, rule.RuleLineno, "Label \"{0}\" for \"{1}({2})\" is never used.", rule.LHSymbolAlias, rule.LHSymbol.Name, rule.LHSymbolAlias);
            /* Generate destructor code for RHS symbols which are not used in the reduce code */
            for (var i = 0; i < rule.RHSymbols.Length; i++)
            {
                if (rule.RHSymbolsAlias[i] != null && !used[i])
                    Context.ErrorMsg(ref ctx.Errors, ctx.Filename, rule.RuleLineno, "Label {0} for \"{1}({2})\" is never used.", rule.RHSymbolsAlias[i], rule.RHSymbols[i].Name, rule.RHSymbolsAlias[i]);
                else if (rule.RHSymbolsAlias[i] == null)
                {
                    if (has_destructor(rule.RHSymbols[i], ctx))
                        b.AppendFormat("  yy_destructor(yypParser,{0},&yymsp[{1}].minor);\n", rule.RHSymbols[i].ID, i - rule.RHSymbols.Length + 1);
                    else { } /* No destructor defined for this term */
                }
            }
            if (rule.Code != null) { rule.Code = (b.ToString() ?? string.Empty); b.Length = 0; }
        }

        /* Generate code which executes when the rule "rp" is reduced.  Write the code to "@out".  Make sure lineno stays up-to-date. */
        private static void emit_code(StreamWriter w, Rule rule, Context ctx, ref int lineno)
        {
            /* Generate code to do the reduce action */
            if (rule.Code != null)
            {
                if (!ctx.NoShowLinenos) { lineno++; Template.tplt_linedir(w, rule.Lineno, ctx.Filename); }
                w.Write("{{{0}", rule.Code);
                var z = rule.Code;
                for (var cp = 0; cp < z.Length; cp++)
                    if (z[cp] == '\n')
                        lineno++;
                w.WriteLine(ref lineno, "}");
                if (!ctx.NoShowLinenos) { lineno++; Template.tplt_linedir(w, lineno, ctx.Outname); }
            }
        }

        /* Print the definition of the union used for the parser's data stack. This union contains fields for every possible data type for tokens
        and nonterminals.  In the process of computing and printing this union, also set the ".dtnum" field of every terminal and nonterminal symbol. */
        private static void print_stack_union(StreamWriter w, Context ctx, ref int lineno, bool makeHeaders)
        {
            /* Build a hash table of datatypes. The ".dtnum" field of each symbol is filled in with the hash index plus 1.  A ".dtnum" value of 0 is
            ** used for terminal symbols.  If there is no %default_type defined then 0 is also used as the .dtnum value for nonterminals which do not specify
            ** a datatype using the %type directive. */
            var dataTypeID = 1;
            var types = new Dictionary<string, int>();
            for (var i = 0; i < ctx.Symbols.Length - 1; i++)
            {
                var symbol = ctx.Symbols[i];
                if (symbol == ctx.ErrorSymbol)
                {
                    symbol.DataTypeID = int.MaxValue;
                    continue;
                }
                if (symbol.Type != SymbolType.NonTerminal || (symbol.DataType == null && ctx.DefaultDataType == null))
                {
                    symbol.DataTypeID = 0;
                    continue;
                }
                var z = (symbol.DataType ?? ctx.DefaultDataType);
                var cp = 0;
                while (char.IsWhiteSpace(z[cp])) cp++;
                var dataType = z.Substring(cp).TrimEnd();
                if (ctx.TokenType != null && string.Equals(dataType, ctx.TokenType))
                {
                    symbol.DataTypeID = 0;
                    continue;
                }
                int value;
                if (types.TryGetValue(dataType, out value))
                    symbol.DataTypeID = value;
                else
                {
                    types.Add(dataType, dataTypeID++);
                    symbol.DataTypeID = dataTypeID;
                }
            }

            /* Print out the definition of YYTOKENTYPE and YYMINORTYPE */
            var name = (ctx.Name ?? "Parse");
            if (makeHeaders) { w.WriteLine(ref lineno, "#if INTERFACE"); }
            w.WriteLine(ref lineno, "#define {0}TOKENTYPE {1}", name, (ctx.TokenType ?? "void*"));
            if (makeHeaders) { w.WriteLine(ref lineno, "#endif"); }
            w.WriteLine(ref lineno, "typedef union {");
            w.WriteLine(ref lineno, "  int yyinit;");
            w.WriteLine(ref lineno, "  {0}TOKENTYPE yy0;", name);
            foreach (var type in types)
                w.WriteLine(ref lineno, "  {0} yy{1};", type.Key, type.Value);
            if (ctx.ErrorSymbol.Uses > 0)
                w.WriteLine(ref lineno, "  int yy{0};", ctx.ErrorSymbol.DataTypeID);
            w.WriteLine(ref lineno, "} YYMINORTYPE;");
        }

        /* Return the name of a C datatype able to represent values between lwr and upr, inclusive. */
        public static string minimum_size_type(int lwr, int upr)
        {
            if (lwr >= 0)
            {
                if (upr <= 255)
                    return "unsigned char";
                else if (upr < 65535)
                    return "unsigned short int";
                else
                    return "unsigned int";
            }
            else if (lwr >= -127 && upr <= 127)
                return "signed char";
            else if (lwr >= -32767 && upr < 32767)
                return "short";
            else
                return "int";
        }

        /* Each state contains a set of token transaction and a set of nonterminal transactions.  Each of these sets makes an instance of the following structure.  An array of these structures is used to order the creation of entries in the yy_action[] table. */
        public struct AX
        {
            public State State;   /* A pointer to a state */
            public bool Token;           /* True to use tokens.  False for non-terminals */
            public int Actions;         /* Number of actions */
            public int iOrder;          /* Original order of action sets */

            public class KeyComparer : IComparer<AX>
            {
                public int Compare(AX x, AX y)
                {
                    if (object.ReferenceEquals(x, y))
                        return 0;
                    var c = y.Actions - x.Actions;
                    if (c == 0)
                        c = y.iOrder - x.iOrder;
                    Debug.Assert(c != 0 || x.Equals(y));
                    return c;
                }
            }
        }

        /* Write text on "w" that describes the rule "rule". */
        private static void WriteRuleText(StreamWriter w, Rule rule)
        {
            w.Write("{0} ::=", rule.LHSymbol.Name);
            for (var j = 0; j < rule.RHSymbols.Length; j++)
            {
                var symbol = rule.RHSymbols[j];
                w.Write(" {0}", symbol.Name);
                if (symbol.Type == SymbolType.MultiTerminal)
                    for (var k = 1; k < symbol.Children.Length; k++)
                        w.Write("|{0}", symbol.Children[k].Name);
            }
        }

        /* Generate C source code for the parser */
        public static void ReportTable(Context ctx, bool makeHeaders) /* Output in makeheaders format if true */
        {
            var filePath = Path.GetFullPath(ctx.Outname ?? ctx.Filename);
            var filePathC = Path.Combine(Path.GetDirectoryName(filePath), Path.GetFileNameWithoutExtension(filePath) + ".c");
            using (var r = Template.tplt_open(ctx))
            using (var w = new StreamWriter(filePathC))
            {
                if (r == null)
                    return;
                if (w == null)
                    return;
                var lineno = 1;
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);
                
                /* Generate the include code, if any */
                Template.tplt_print(w, ctx, ctx.Include, ref lineno);
                if (makeHeaders)
                {
                    var filePathH = Path.Combine(Path.GetDirectoryName(filePath), Path.GetFileNameWithoutExtension(filePath) + ".h");
                    w.WriteLine(ref lineno, "#include \"{0}\"", filePathH);
                }
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate #defines for all tokens */
                if (makeHeaders)
                {
                    w.WriteLine(ref lineno, "#if INTERFACE");
                    var prefix = (ctx.TokenPrefix ?? string.Empty);
                    for (var i = 1; i < ctx.Terminals; i++)
                        w.WriteLine(ref lineno, "#define {0}{1,-30} {2,2}", prefix, ctx.Symbols[i].Name, i);
                    w.WriteLine(ref lineno, "#endif");
                }
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate the defines */
                w.WriteLine(ref lineno, "#define YYCODETYPE {0}", minimum_size_type(0, ctx.Symbols.Length));
                w.WriteLine(ref lineno, "#define YYNOCODE {0}", ctx.Symbols.Length);
                w.WriteLine(ref lineno, "#define YYACTIONTYPE {0}", minimum_size_type(0, ctx.States + ctx.Rules + 5));
                if (ctx.Wildcard != null)
                    w.WriteLine(ref lineno, "#define YYWILDCARD {0}", ctx.Wildcard.ID);
                print_stack_union(w, ctx, ref lineno, makeHeaders);
                w.WriteLine(ref lineno, "#ifndef YYSTACKDEPTH");
                if (ctx.StackSize != null)
                    w.WriteLine(ref lineno, "#define YYSTACKDEPTH {0}", ctx.StackSize);
                else
                    w.WriteLine(ref lineno, "#define YYSTACKDEPTH 100");
                w.WriteLine(ref lineno, "#endif");
                if (makeHeaders)
                    w.WriteLine(ref lineno, "#if INTERFACE");
                var name = (ctx.Name ?? "Parse");
                if (!string.IsNullOrEmpty(ctx.ExtraArg))
                {
                    var i = ctx.ExtraArg.Length;
                    while (i >= 1 && char.IsWhiteSpace(ctx.ExtraArg[i - 1])) i--;
                    while (i >= 1 && (char.IsLetterOrDigit(ctx.ExtraArg[i - 1]) || ctx.ExtraArg[i - 1] == '_')) i--;
                    w.WriteLine(ref lineno, "#define {0}ARG_SDECL {1};", name, ctx.ExtraArg);
                    w.WriteLine(ref lineno, "#define {0}ARG_PDECL ,{1}", name, ctx.ExtraArg);
                    w.WriteLine(ref lineno, "#define {0}ARG_FETCH {1} = yypParser.{2}", name, ctx.ExtraArg, ctx.ExtraArg.Substring(i));
                    w.WriteLine(ref lineno, "#define {0}ARG_STORE yypParser.{1} = {2}", name, ctx.ExtraArg.Substring(i), ctx.ExtraArg.Substring(i));
                }
                else
                {
                    w.WriteLine(ref lineno, "#define {0}ARG_SDECL", name);
                    w.WriteLine(ref lineno, "#define {0}ARG_PDECL", name);
                    w.WriteLine(ref lineno, "#define {0}ARG_FETCH", name);
                    w.WriteLine(ref lineno, "#define {0}ARG_STORE", name);
                }
                if (makeHeaders)
                    w.WriteLine(ref lineno, "#endif");
                w.WriteLine(ref lineno, "#define YYNSTATE {0}", ctx.States);
                w.WriteLine(ref lineno, "#define YYNRULE {0}", ctx.Rules);
                if (ctx.ErrorSymbol.Uses > 0)
                {
                    w.WriteLine(ref lineno, "#define YYERRORSYMBOL {0}", ctx.ErrorSymbol.ID);
                    w.WriteLine(ref lineno, "#define YYERRSYMDT yy{0}", ctx.ErrorSymbol.DataTypeID);
                }
                if (ctx.HasFallback)
                    w.WriteLine(ref lineno, "#define YYFALLBACK 1");
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate the action table and its associates:
                **
                **  yy_action[]        A single table containing all actions.
                **  yy_lookahead[]     A table containing the lookahead for each entry in
                **                     yy_action.  Used to detect hash collisions.
                **  yy_shift_ofst[]    For each state, the offset into yy_action for
                **                     shifting terminals.
                **  yy_reduce_ofst[]   For each state, the offset into yy_action for
                **                     shifting non-terminals after a reduce.
                **  yy_default[]       Default action for each state.
                */

                /* Compute the actions on all states and count them up */
                var ax = new AX[ctx.States * 2];
                for (var i = 0; i < ctx.States; i++)
                {
                    var state = ctx.Sorted[i];
                    ax[i * 2] = new AX
                    {
                        State = state,
                        Token = true,
                        Actions = state.TokenActions,
                    };
                    ax[i * 2 + 1] = new AX
                    {
                        State = state,
                        Token = false,
                        Actions = state.TokenActions,
                    };
                }
                var maxTokenOffset = 0;
                var minTokenOffset = 0;
                var maxNonTerminalOffset = 0;
                var minNonTerminalOffset = 0;
                /* Compute the action table.  In order to try to keep the size of the action table to a minimum, the heuristic of placing the largest action sets first is used. */
                for (var i = 0; i < ctx.States * 2; i++)
                    ax[i].iOrder = i;
                Array.Sort(ax, _keyComparer);
                var actionTable = new ActionTable();
                for (var i = 0; i < ctx.States * 2 && ax[i].Actions > 0; i++)
                {
                    var state = ax[i].State;
                    if (ax[i].Token)
                    {
                        foreach (var action in state.Actions)
                        {
                            if (action.Symbol.ID >= ctx.Terminals) continue;
                            var actionID = action.ComputeID(ctx);
                            if (actionID < 0) continue;
                            actionTable.Action(action.Symbol.ID, actionID);
                        }
                        state.TokenOffset = actionTable.Insert();
                        if (state.TokenOffset < minTokenOffset) minTokenOffset = state.TokenOffset;
                        if (state.TokenOffset > maxTokenOffset) maxTokenOffset = state.TokenOffset;
                    }
                    else
                    {
                        foreach (var action in state.Actions)
                        {
                            if (action.Symbol.ID < ctx.Terminals) continue;
                            if (action.Symbol.ID == ctx.Symbols.Length - 1) continue;
                            var actionID = action.ComputeID(ctx);
                            if (actionID < 0) continue;
                            actionTable.Action(action.Symbol.ID, actionID);
                        }
                        state.NonTerminalOffset = actionTable.Insert();
                        if (state.NonTerminalOffset < minNonTerminalOffset) minNonTerminalOffset = state.NonTerminalOffset;
                        if (state.NonTerminalOffset > maxNonTerminalOffset) maxNonTerminalOffset = state.NonTerminalOffset;
                    }
                }
                ax = null;
                /* Output the yy_action table */
                var n = actionTable.Size;
                w.WriteLine(ref lineno, "#define YY_ACTTAB_COUNT ({0})", n);
                w.WriteLine(ref lineno, "static const YYACTIONTYPE yy_action[] = {");
                for (int i = 0, j = 0; i < n; i++)
                {
                    var action = actionTable.GetAction(i);
                    if (action < 0)
                        action = ctx.States + ctx.Rules + 2;
                    if (j == 0)
                        w.Write(" /* {0,5} */ ", i);
                    w.Write(" {0,4},", action);
                    if (j == 9 || i == n - 1)
                    {
                        w.WriteLine(ref lineno);
                        j = 0;
                    }
                    else
                        j++;
                }
                w.WriteLine(ref lineno, "};");

                /* Output the yy_lookahead table */
                w.WriteLine(ref lineno, "static const YYCODETYPE yy_lookahead[] = {");
                for (int i = 0, j = 0; i < n; i++)
                {
                    var lookahead = actionTable.GetLookahead(i);
                    if (lookahead < 0)
                        lookahead = ctx.Symbols.Length - 1;
                    if (j == 0)
                        w.Write(" /* {0,5} */ ", i);
                    w.Write(" {0,4},", lookahead);
                    if (j == 9 || i == n - 1)
                    {
                        w.WriteLine(ref lineno);
                        j = 0;
                    }
                    else
                        j++;
                }
                w.WriteLine(ref lineno, "};");

                /* Output the yy_shift_ofst[] table */
                w.WriteLine(ref lineno, "#define YY_SHIFT_USE_DFLT ({0})", minTokenOffset - 1);
                n = ctx.States;
                while (n > 0 && ctx.Sorted[n - 1].TokenOffset == State.NO_OFFSET) n--;
                w.WriteLine(ref lineno, "#define YY_SHIFT_COUNT ({0})", n - 1);
                w.WriteLine(ref lineno, "#define YY_SHIFT_MIN   ({0})", minTokenOffset);
                w.WriteLine(ref lineno, "#define YY_SHIFT_MAX   ({0})", maxTokenOffset);
                w.WriteLine(ref lineno, "static const {0} yy_shift_ofst[] = {", minimum_size_type(minTokenOffset - 1, maxTokenOffset));
                for (int i = 0, j = 0; i < n; i++)
                {
                    var state = ctx.Sorted[i];
                    var offset = state.TokenOffset;
                    if (offset == State.NO_OFFSET) offset = minTokenOffset - 1;
                    if (j == 0) w.Write(" /* {0,5} */ ", i);
                    w.Write(" {0,4},", offset);
                    if (j == 9 || i == n - 1)
                    {
                        w.WriteLine(ref lineno);
                        j = 0;
                    }
                    else
                        j++;
                }
                w.WriteLine(ref lineno, "};");

                /* Output the yy_reduce_ofst[] table */
                w.WriteLine(ref lineno, "#define YY_REDUCE_USE_DFLT ({0})", minNonTerminalOffset - 1);
                n = ctx.States;
                while (n > 0 && ctx.Sorted[n - 1].NonTerminalOffset == State.NO_OFFSET) n--;
                w.WriteLine(ref lineno, "#define YY_REDUCE_COUNT ({0})", n - 1);
                w.WriteLine(ref lineno, "#define YY_REDUCE_MIN   ({0})", minNonTerminalOffset);
                w.WriteLine(ref lineno, "#define YY_REDUCE_MAX   ({0})", maxNonTerminalOffset);
                w.WriteLine(ref lineno, "static const {0} yy_reduce_ofst[] = {", minimum_size_type(minNonTerminalOffset - 1, maxNonTerminalOffset));
                for (int i = 0, j = 0; i < n; i++)
                {
                    var state = ctx.Sorted[i];
                    var offset = state.NonTerminalOffset;
                    if (offset == State.NO_OFFSET) offset = minNonTerminalOffset - 1;
                    if (j == 0) w.Write(" /* {0,5} */ ", i);
                    w.Write(" {0,4},", offset);
                    if (j == 9 || i == n - 1)
                    {
                        w.WriteLine(ref lineno);
                        j = 0;
                    }
                    else
                        j++;
                }
                w.WriteLine(ref lineno, "};");

                /* Output the default action table */
                w.WriteLine(ref lineno, "static const YYACTIONTYPE yy_default[] = {");
                n = ctx.States;
                for (int i = 0, j = 0; i < n; i++)
                {
                    var stp = ctx.Sorted[i];
                    if (j == 0) w.Write(" /* {0,5} */ ", i);
                    w.Write(" {0,4},", stp.Default);
                    if (j == 9 || i == n - 1)
                    {
                        w.WriteLine(ref lineno);
                        j = 0;
                    }
                    else
                        j++;
                }
                w.WriteLine(ref lineno, "};");
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate the table of fallback tokens. */
                if (ctx.HasFallback)
                {
                    var mx = ctx.Terminals - 1;
                    while (mx > 0 && ctx.Symbols[mx].Fallback == null) { mx--; }
                    for (var i = 0; i <= mx; i++)
                    {
                        var symbol = ctx.Symbols[i];
                        if (symbol.Fallback == null)
                            w.WriteLine(ref lineno, "    0,  /* {0,10} => nothing */", symbol.Name);
                        else
                            w.WriteLine(ref lineno, "  {0,3},  /* {1,10} => {2} */", symbol.Fallback.ID, symbol.Name, symbol.Fallback.Name);
                    }
                }
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate a table containing the symbolic name of every symbol */
                var i_ = 0;
                for (i_ = 0; i_ < ctx.Symbols.Length - 1; i_++)
                {
                    w.Write("  {0,-15}", string.Format("\"{0}\",", ctx.Symbols[i_].Name));
                    if ((i_ & 3) == 3) { w.WriteLine(ref lineno); }
                }
                if ((i_ & 3) != 0) { w.WriteLine(ref lineno); }
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate a table containing a text string that describes every rule in the rule set of the grammar.  This information is used when tracing REDUCE actions. */
                i_ = 0;
                for (var rule = ctx.Rule; rule != null; rule = rule.Next, i_++)
                {
                    Debug.Assert(rule.ID == i_);
                    w.Write(" /* {0,3} */ \"", i_);
                    WriteRuleText(w, rule);
                    w.WriteLine(ref lineno, "\",");
                }
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate code which executes every time a symbol is popped from the stack while processing errors or while destroying the parser.  (In other words, generate the %destructor actions) */
                if (ctx.TokenDestructor != null)
                {
                    var once = true;
                    for (var i = 0; i < ctx.Symbols.Length - 1; i++)
                    {
                        var symbol = ctx.Symbols[i];
                        if (symbol == null || symbol.Type != SymbolType.Terminal) continue;
                        if (once)
                        {
                            w.WriteLine(ref lineno, "      /* TERMINAL Destructor */");
                            once = false;
                        }
                        w.WriteLine(ref lineno, "    case {0}: /* {1} */", symbol.ID, symbol.Name);
                    }
                    var i2 = 0;
                    for (; i2 < (ctx.Symbols.Length - 1) && (ctx.Symbols[i2].Type != SymbolType.Terminal); i2++) ;
                    if (i2 < ctx.Symbols.Length - 1)
                    {
                        emit_destructor_code(w, ctx.Symbols[i2], ctx, ref lineno);
                        w.WriteLine(ref lineno, "      break;");
                    }
                }
                if (ctx.DefaultDestructor != null)
                {
                    var dflt_sp = (Symbol)null;
                    var once = true;
                    for (var i = 0; i < ctx.Symbols.Length - 1; i++)
                    {
                        var sp = ctx.Symbols[i];
                        if (sp == null || sp.Type == SymbolType.Terminal || sp.ID <= 0 || sp.Destructor != null) continue;
                        if (once)
                        {
                            w.WriteLine(ref lineno, "      /* Default NON-TERMINAL Destructor */");
                            once = false;
                        }
                        w.WriteLine(ref lineno, "    case {0}: /* {1} */", sp.ID, sp.Name);
                        dflt_sp = sp;
                    }
                    if (dflt_sp != null)
                        emit_destructor_code(w, dflt_sp, ctx, ref lineno);
                    w.WriteLine(ref lineno, "      break;");
                }
                for (var i = 0; i < ctx.Symbols.Length - 1; i++)
                {
                    var symbol = ctx.Symbols[i];
                    if (symbol == null || symbol.Type == SymbolType.Terminal || symbol.Destructor == null) continue;
                    w.WriteLine(ref lineno, "    case {0}: /* {1} */", symbol.ID, symbol.Name);
                    /* Combine duplicate destructors into a single case */
                    for (var j = i + 1; j < ctx.Symbols.Length - 1; j++)
                    {
                        var sp2 = ctx.Symbols[j];
                        if (sp2 != null && sp2.Type != SymbolType.Terminal && sp2.Destructor != null && sp2.DataTypeID == symbol.DataTypeID && symbol.Destructor == sp2.Destructor)
                        {
                            w.WriteLine(ref lineno, "    case {0}: /* {1} */", sp2.ID, sp2.Name);
                            sp2.Destructor = null;
                        }
                    }
                    emit_destructor_code(w, ctx.Symbols[i], ctx, ref lineno);
                    w.WriteLine(ref lineno, "      break;");
                }
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate code which executes whenever the parser stack overflows */
                Template.tplt_print(w, ctx, ctx.StackOverflow, ref lineno);
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate the table of rule information 
                ** Note: This code depends on the fact that rules are number sequentually beginning with 0. */
                for (var rp = ctx.Rule; rp != null; rp = rp.Next)
                    w.WriteLine(ref lineno, "  { {0}, {0} },", rp.LHSymbol.ID, rp.RHSymbols.Length);
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate code which execution during each REDUCE action */
                for (var rp = ctx.Rule; rp != null; rp = rp.Next)
                    translate_code(ctx, rp);
                /* First output rules other than the default: rule */
                for (var rp = ctx.Rule; rp != null; rp = rp.Next)
                {
                    if (rp.Code == null) continue;
                    //TODO:Sky this will break
                    if (rp.Code[0] == '\n' && rp.Code[1] == 0) continue; /* Will be default: */
                    w.Write("      case {0}: /* ", rp.ID);
                    WriteRuleText(w, rp);
                    w.WriteLine(ref lineno, " */");
                    for (var rp2 = rp.Next; rp2 != null; rp2 = rp2.Next)
                        if (rp2.Code == rp.Code)
                        {
                            w.Write("      case {0}: /* ", rp2.ID);
                            WriteRuleText(w, rp2);
                            w.WriteLine(ref lineno, " */ yytestcase(yyruleno=={0});", rp2.ID);
                            rp2.Code = null;
                        }
                    emit_code(w, rp, ctx, ref lineno);
                    w.WriteLine(ref lineno, "        break;");
                    rp.Code = null;
                }
                /* Finally, output the default: rule.  We choose as the default: all empty actions. */
                w.WriteLine(ref lineno, "      default:");
                for (var rp = ctx.Rule; rp != null; rp = rp.Next)
                {
                    if (rp.Code == null) continue;
                    //TODO:Sky this will break
                    Debug.Assert(rp.Code[0] == '\n' && rp.Code[1] == 0);
                    w.Write("      /* ({0}) ", rp.ID);
                    WriteRuleText(w, rp);
                    w.WriteLine(ref lineno, " */ yytestcase(yyruleno=={0});", rp.ID);
                }
                w.WriteLine(ref lineno, "        break;");
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate code which executes if a parse fails */
                Template.tplt_print(w, ctx, ctx.ParseFailure, ref lineno);
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate code which executes when a syntax error occurs */
                Template.tplt_print(w, ctx, ctx.SyntaxError, ref lineno);
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Generate code which executes when the parser accepts its input */
                Template.tplt_print(w, ctx, ctx.ParseAccept, ref lineno);
                Template.tplt_xfer(ctx.Name, r, w, ref lineno);

                /* Append any addition code the user desires */
                Template.tplt_print(w, ctx, ctx.ExtraCode, ref lineno);
            }
        }

        /* Generate a header file for the parser */
        public static void ReportHeader(Context ctx)
        {
            var prefix = (ctx.TokenPrefix ?? string.Empty);
            var filePath = Path.GetFileNameWithoutExtension(ctx.Outname ?? ctx.Filename) + ".h";
            if (File.Exists(filePath))
                using (var fp = new StreamReader(filePath))
                {
                    string line = null;
                    var i = 1;
                    for (; i < ctx.Terminals && (line = fp.ReadLine()) != null; i++)
                        if (line == string.Format("#define {0}{1,-30} {2,2}\n", prefix, ctx.Symbols[i].Name, i))
                            break;
                    /* No change in the file.  Don't rewrite it. */
                    if (i == ctx.Terminals)
                        return;
                }
            using (var fp = new StreamWriter(filePath))
                for (var i = 1; i < ctx.Terminals; i++)
                    fp.Write("#define {0}{1,-30} {2,2}\n", prefix, ctx.Symbols[i].Name, i);
        }
    }
}
