using System;
using System.IO;

namespace Lemon
{
    public class Parser
    {
        public enum FiniteState
        {
            INITIALIZE,
            WAITING_FOR_DECL_OR_RULE,
            WAITING_FOR_DECL_KEYWORD,
            WAITING_FOR_DECL_ARG,
            WAITING_FOR_PRECEDENCE_SYMBOL,
            WAITING_FOR_ARROW,
            IN_RHS,
            LHS_ALIAS_1,
            LHS_ALIAS_2,
            LHS_ALIAS_3,
            RHS_ALIAS_1,
            RHS_ALIAS_2,
            PRECEDENCE_MARK_1,
            PRECEDENCE_MARK_2,
            RESYNC_AFTER_RULE_ERROR,
            RESYNC_AFTER_DECL_ERROR,
            WAITING_FOR_DESTRUCTOR_SYMBOL,
            WAITING_FOR_DATATYPE_SYMBOL,
            WAITING_FOR_FALLBACK_ID,
            WAITING_FOR_WILDCARD_ID
        }

        public class ParserContext
        {
            public const int MAXRHS = 100;
            public string Filename;
            public int TokenLineno;
            public int Errors;
            public int TokenStart;
            public int TokenEnd;
            public FiniteState FiniteState;
            public Symbol FallbackSymbol;
            public Symbol LHSymbol;
            public string LHSymbolAlias;
            public int QueuedRHSymbols;
            public Symbol[] QueuedRHSymbol = new Symbol[MAXRHS];
            public string[] QueuedRHSymbolAlias = new string[MAXRHS];
            public Rule PrevRule;
            public string DeclKeyword;
            public Accessor<string>? DeclArgSlot;
            public bool InsertLineMacro;
            public Accessor<int>? DeclLinenoSlot;
            public Association DeclAssociation;
            public int Precedences;
            public Rule FirstRule;
            public Rule LastRule;
        }

        public struct Accessor<T>
        {
            public T v;
            public Func<T> Get;
            public Action<T> Set;
        }

        private static void ParseSingleToken(string z, Context ctx, ParserContext pctx)
        {
            var v = z.Substring(pctx.TokenStart, pctx.TokenEnd - pctx.TokenStart);
#if false
            Console.WriteLine("{0}:{1}: Token=[{2}] state={3}", pctx.Filename, pctx.TokenLineno, v, pctx.FiniteState);
#endif
            switch (pctx.FiniteState)
            {
                case FiniteState.INITIALIZE:
                case FiniteState.WAITING_FOR_DECL_OR_RULE:
                    if (pctx.FiniteState == FiniteState.INITIALIZE)
                    {
                        pctx.PrevRule = pctx.FirstRule = pctx.LastRule = null;
                        pctx.Precedences = 0;
                        ctx.Rules = 0;
                    }
                    //
                    if (v[0] == '%')
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_KEYWORD;
                    else if (char.IsLower(v[0]))
                    {
                        pctx.LHSymbol = Symbol.New(v);
                        pctx.QueuedRHSymbols = 0;
                        pctx.LHSymbolAlias = null;
                        pctx.FiniteState = FiniteState.WAITING_FOR_ARROW;
                    }
                    else if (v[0] == '{')
                    {
                        if (pctx.PrevRule == null)
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "There is no prior rule opon which to attach the code fragment which begins on this line.");
                        else if (pctx.PrevRule.Code != null)
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Code fragment beginning on this line is not the first to follow the previous rule.");
                        else
                        {
                            pctx.PrevRule.Lineno = pctx.TokenLineno;
                            pctx.PrevRule.Code = v.Substring(1);
                        }
                    }
                    else if (v[0] == '[')
                        pctx.FiniteState = FiniteState.PRECEDENCE_MARK_1;
                    else
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Token \"{0}\" should be either \"%\" or a nonterminal name.", v);
                    break;
                case FiniteState.PRECEDENCE_MARK_1:
                    if (!char.IsUpper(v[0]))
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "The precedence symbol must be a terminal.");
                    else if (pctx.PrevRule == null)
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "There is no prior rule to assign precedence \"[{0}]\".", v);
                    else if (pctx.PrevRule.PrecedenceSymbol != null)
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Precedence mark on this line is not the first to follow the previous rule.");
                    else
                        pctx.PrevRule.PrecedenceSymbol = Symbol.New(v);
                    pctx.FiniteState = FiniteState.PRECEDENCE_MARK_2;
                    break;
                case FiniteState.PRECEDENCE_MARK_2:
                    if (v[0] != ']')
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Missing \"]\" on precedence mark.");
                    pctx.FiniteState = FiniteState.WAITING_FOR_DECL_OR_RULE;
                    break;
                case FiniteState.WAITING_FOR_ARROW:
                    if (v[0] == ':' && v[1] == ':' && v[2] == '=')
                        pctx.FiniteState = FiniteState.IN_RHS;
                    else if (v[0] == '(')
                        pctx.FiniteState = FiniteState.LHS_ALIAS_1;
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Expected to see a \":\" following the LHS symbol \"{0}\".", pctx.LHSymbol.Name);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case FiniteState.LHS_ALIAS_1:
                    if (char.IsLetter(v[0]))
                    {
                        pctx.LHSymbolAlias = v;
                        pctx.FiniteState = FiniteState.LHS_ALIAS_2;
                    }
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "\"{0}\" is not a valid alias for the LHS \"{1}\"\n", v, pctx.LHSymbol.Name);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case FiniteState.LHS_ALIAS_2:
                    if (v[0] == ')')
                        pctx.FiniteState = FiniteState.LHS_ALIAS_3;
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Missing \")\" following LHS alias name \"{0}\".", pctx.LHSymbolAlias);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case FiniteState.LHS_ALIAS_3:
                    if (v[0] == ':' && v[1] == ':' && v[2] == '=')
                        pctx.FiniteState = FiniteState.IN_RHS;
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Missing \".\" following: \"{0}({1})\".", pctx.LHSymbol.Name, pctx.LHSymbolAlias);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case FiniteState.IN_RHS:
                    if (v[0] == '.')
                    {
                        var rule = new Rule
                        {
                            RuleLineno = pctx.TokenLineno,
                            RHSymbols = new Symbol[pctx.QueuedRHSymbols],
                            RHSymbolsAlias = new string[pctx.QueuedRHSymbols],
                            LHSymbol = pctx.LHSymbol,
                            LHSymbolAlias = pctx.LHSymbolAlias,
                            ID = ctx.Rules++,
                        };
                        for (var index = 0; index < pctx.QueuedRHSymbols; index++)
                        {
                            rule.RHSymbols[index] = pctx.QueuedRHSymbol[index];
                            rule.RHSymbolsAlias[index] = pctx.QueuedRHSymbolAlias[index];
                        }
                        rule.NextLHSymbol = rule.LHSymbol.Rule;
                        rule.LHSymbol.Rule = rule;
                        rule.Next = null;
                        if (pctx.FirstRule == null)
                            pctx.FirstRule = pctx.LastRule = rule;
                        else
                        {
                            pctx.LastRule.Next = rule;
                            pctx.LastRule = rule;
                        }
                        pctx.PrevRule = rule;
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_OR_RULE;
                    }
                    else if (char.IsLetter(v[0]))
                    {
                        if (pctx.QueuedRHSymbols >= ParserContext.MAXRHS)
                        {
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Too many symbols on RHS of rule beginning at \"{0}\".", v);
                            pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                        }
                        else
                        {
                            pctx.QueuedRHSymbol[pctx.QueuedRHSymbols] = Symbol.New(v);
                            pctx.QueuedRHSymbolAlias[pctx.QueuedRHSymbols] = null;
                            pctx.QueuedRHSymbols++;
                        }
                    }
                    else if ((v[0] == '|' || v[0] == '/') && pctx.QueuedRHSymbols > 0)
                    {
                        var multiSymbol = pctx.QueuedRHSymbol[pctx.QueuedRHSymbols - 1];
                        if (multiSymbol.Type != SymbolType.MultiTerminal)
                        {
                            var originalSymbol = multiSymbol;
                            multiSymbol = new Symbol
                            {
                                Type = SymbolType.MultiTerminal,
                                Children = new[] { originalSymbol },
                                Name = originalSymbol.Name,
                            };
                            pctx.QueuedRHSymbol[pctx.QueuedRHSymbols - 1] = multiSymbol;
                        }
                        Array.Resize(ref multiSymbol.Children, multiSymbol.Children.Length + 1);
                        multiSymbol.Children[multiSymbol.Children.Length - 1] = Symbol.New(v.Substring(1));
                        if (char.IsLower(v[1]) || char.IsLower(multiSymbol.Children[0].Name[0]))
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Cannot form a compound containing a non-terminal");
                    }
                    else if (v[0] == '(' && pctx.QueuedRHSymbols > 0)
                        pctx.FiniteState = FiniteState.RHS_ALIAS_1;
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Illegal character on RHS of rule: \"{0}\".", v);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case FiniteState.RHS_ALIAS_1:
                    if (char.IsLetter(v[0]))
                    {
                        pctx.QueuedRHSymbolAlias[pctx.QueuedRHSymbols - 1] = v;
                        pctx.FiniteState = FiniteState.RHS_ALIAS_2;
                    }
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "\"{0}\" is not a valid alias for the RHS symbol \"{1}\"\n", v, pctx.QueuedRHSymbol[pctx.QueuedRHSymbols - 1].Name);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case FiniteState.RHS_ALIAS_2:
                    if (v[0] == ')')
                        pctx.FiniteState = FiniteState.IN_RHS;
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Missing \")\" following LHS alias name \"{0}\".", pctx.LHSymbolAlias);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case FiniteState.WAITING_FOR_DECL_KEYWORD:
                    if (char.IsLetter(v[0]))
                    {
                        pctx.DeclKeyword = v;
                        pctx.DeclArgSlot = null;
                        pctx.DeclLinenoSlot = null;
                        pctx.InsertLineMacro = true;
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_ARG;
                        if (v == "name")
                        {
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.Name, Set = x => ctx.Name = x };
                            pctx.InsertLineMacro = false;
                        }
                        else if (v == "include")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.Include, Set = x => ctx.Include = x };
                        else if (v == "code")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.ExtraCode, Set = x => ctx.ExtraCode = x };
                        else if (v == "token_destructor")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.TokenDestructor, Set = x => ctx.TokenDestructor = x };
                        else if (v == "default_destructor")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.DefaultDestructor, Set = x => ctx.DefaultDestructor = x };
                        else if (v == "token_prefix")
                        {
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.TokenPrefix, Set = x => ctx.TokenPrefix = x };
                            pctx.InsertLineMacro = false;
                        }
                        else if (v == "syntax_error")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.SyntaxError, Set = x => ctx.SyntaxError = x };
                        else if (v == "parse_accept")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.ParseAccept, Set = x => ctx.ParseAccept = x };
                        else if (v == "parse_failure")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.ParseFailure, Set = x => ctx.ParseFailure = x };
                        else if (v == "stack_overflow")
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.StackOverflow, Set = x => ctx.StackOverflow = x };
                        else if (v == "extra_argument")
                        {
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.ExtraArg, Set = x => ctx.ExtraArg = x };
                            pctx.InsertLineMacro = false;
                        }
                        else if (v == "token_type")
                        {
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.TokenType, Set = x => ctx.TokenType = x };
                            pctx.InsertLineMacro = false;
                        }
                        else if (v == "default_type")
                        {
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.DefaultDataType, Set = x => ctx.DefaultDataType = x };
                            pctx.InsertLineMacro = false;
                        }
                        else if (v == "stack_size")
                        {
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.StackSize, Set = x => ctx.StackSize = x };
                            pctx.InsertLineMacro = false;
                        }
                        else if (v == "start_symbol")
                        {
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => ctx.StartSymbol, Set = x => ctx.StartSymbol = x };
                            pctx.InsertLineMacro = false;
                        }
                        else if (v == "left")
                        {
                            pctx.Precedences++;
                            pctx.DeclAssociation = Association.Left;
                            pctx.FiniteState = FiniteState.WAITING_FOR_PRECEDENCE_SYMBOL;
                        }
                        else if (v == "right")
                        {
                            pctx.Precedences++;
                            pctx.DeclAssociation = Association.Right;
                            pctx.FiniteState = FiniteState.WAITING_FOR_PRECEDENCE_SYMBOL;
                        }
                        else if (v == "nonassoc")
                        {
                            pctx.Precedences++;
                            pctx.DeclAssociation = Association.None;
                            pctx.FiniteState = FiniteState.WAITING_FOR_PRECEDENCE_SYMBOL;
                        }
                        else if (v == "destructor")
                            pctx.FiniteState = FiniteState.WAITING_FOR_DESTRUCTOR_SYMBOL;
                        else if (v == "type")
                            pctx.FiniteState = FiniteState.WAITING_FOR_DATATYPE_SYMBOL;
                        else if (v == "fallback")
                        {
                            pctx.FallbackSymbol = null;
                            pctx.FiniteState = FiniteState.WAITING_FOR_FALLBACK_ID;
                        }
                        else if (v == "wildcard")
                            pctx.FiniteState = FiniteState.WAITING_FOR_WILDCARD_ID;
                        else
                        {
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Unknown declaration keyword: \"%{0}\".", v);
                            pctx.FiniteState = FiniteState.RESYNC_AFTER_DECL_ERROR;
                        }
                    }
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Illegal declaration keyword: \"{0}\".", v);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_DECL_ERROR;
                    }
                    break;
                case FiniteState.WAITING_FOR_DESTRUCTOR_SYMBOL:
                    if (!char.IsLetter(v[0]))
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Symbol name missing after %destructor keyword");
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_DECL_ERROR;
                    }
                    else
                    {
                        var symbol = Symbol.New(v);
                        pctx.DeclArgSlot = new Accessor<string> { Get = () => symbol.Destructor, Set = x => symbol.Destructor = x };
                        pctx.DeclLinenoSlot = new Accessor<int> { Get = () => symbol.DestructorLineno, Set = x => symbol.DestructorLineno = x };
                        pctx.InsertLineMacro = true;
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_ARG;
                    }
                    break;
                case FiniteState.WAITING_FOR_DATATYPE_SYMBOL:
                    if (!char.IsLetter(v[0]))
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Symbol name missing after %type keyword");
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_DECL_ERROR;
                    }
                    else
                    {
                        Symbol symbol;
                        if (!Context.AllSymbols.TryGetValue(v, out symbol))
                            symbol = null;
                        if (symbol != null && symbol.DataType != null)
                        {
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Symbol %type \"{0}\" already defined", v);
                            pctx.FiniteState = FiniteState.RESYNC_AFTER_DECL_ERROR;
                        }
                        else
                        {
                            if (symbol == null)
                                symbol = Symbol.New(v);
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => symbol.DataType, Set = x => symbol.DataType = x };
                            pctx.InsertLineMacro = false;
                            pctx.FiniteState = FiniteState.WAITING_FOR_DECL_ARG;
                        }
                    }
                    break;
                case FiniteState.WAITING_FOR_PRECEDENCE_SYMBOL:
                    if (v[0] == '.')
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_OR_RULE;
                    else if (char.IsLetter(v[0]))
                    {
                        var symbol = Symbol.New(v);
                        if (symbol.Precedence >= 0)
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Symbol \"{0}\" has already be given a precedence.", v);
                        else
                        {
                            symbol.Precedence = pctx.Precedences;
                            symbol.Association = pctx.DeclAssociation;
                        }
                    }
                    else
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Can't assign a precedence to \"{0}\".", v);
                    break;
                case FiniteState.WAITING_FOR_DECL_ARG:
                    if (v[0] == '{' || v[0] == '"' || char.IsLetterOrDigit(v[0]))
                    {
                        var zNew = (v[0] == '{' || v[0] == '"' ? v.Substring(1) : v);
                        if (!pctx.DeclArgSlot.HasValue)
                            pctx.DeclArgSlot = new Accessor<string> { Get = () => v, Set = x => v = x };
                        var zBuf = pctx.DeclArgSlot.Value.Get();
                        var addLineMacro = !ctx.NoShowLinenos && pctx.InsertLineMacro && (!pctx.DeclLinenoSlot.HasValue || pctx.DeclLinenoSlot.Value.Get() != 0);
                        if (addLineMacro)
                        {
                            if (!string.IsNullOrEmpty(zBuf) && !zBuf.EndsWith("\n")) zBuf += "\n";
                            zBuf += string.Format("#line {0} \"{1}\"\n", pctx.TokenLineno, pctx.Filename.Replace("\\", "\\\\"));
                        }
                        if (pctx.DeclLinenoSlot.HasValue && pctx.DeclLinenoSlot.Value.Get() == 0)
                            pctx.DeclLinenoSlot.Value.Set(pctx.TokenLineno);
                        zBuf += zNew;
                        pctx.DeclArgSlot.Value.Set(zBuf);
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_OR_RULE;
                    }
                    else
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Illegal argument to %{0}: {1}", pctx.DeclKeyword, v);
                        pctx.FiniteState = FiniteState.RESYNC_AFTER_DECL_ERROR;
                    }
                    break;
                case FiniteState.WAITING_FOR_FALLBACK_ID:
                    if (v[0] == '.')
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_OR_RULE;
                    else if (!char.IsUpper(v[0]))
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "%fallback argument \"{0}\" should be a token", v);
                    else
                    {
                        var symbol = Symbol.New(v);
                        if (pctx.FallbackSymbol == null)
                            pctx.FallbackSymbol = symbol;
                        else if (symbol.Fallback != null)
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "More than one fallback assigned to token {0}", v);
                        else
                        {
                            symbol.Fallback = pctx.FallbackSymbol;
                            ctx.HasFallback = true;
                        }
                    }
                    break;
                case FiniteState.WAITING_FOR_WILDCARD_ID:
                    if (v[0] == '.')
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_OR_RULE;
                    else if (!char.IsUpper(v[0]))
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "%wildcard argument \"{0}\" should be a token", v);
                    else
                    {
                        var symbol = Symbol.New(v);
                        if (ctx.Wildcard == null)
                            ctx.Wildcard = symbol;
                        else
                            Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "Extra wildcard to token: {0}", v);
                    }
                    break;
                case FiniteState.RESYNC_AFTER_RULE_ERROR:
                case FiniteState.RESYNC_AFTER_DECL_ERROR:
                    if (v[0] == '.')
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_OR_RULE;
                    if (v[0] == '%')
                        pctx.FiniteState = FiniteState.WAITING_FOR_DECL_KEYWORD;
                    break;
            }
        }

        private static void PreProcessInput(ref string z, string[] defines)
        {
            if (defines == null)
                throw new ArgumentNullException("defines");
            var exclude = 0;
            var start = 0;
            var lineno = 1;
            var start_lineno = 1;
            var removeCount = 0;
            for (var i = 0; i < z.Length; i++)
            {
                if (z[i] == '\n') lineno++;
                if (z[i] != '%' || (i > 0 && (z[i - 1] != '\n'))) continue;
                if (z.Substring(i, 6) == "%endif" && char.IsWhiteSpace(z[i + 6]))
                {
                    if (exclude > 0)
                    {
                        exclude--;
                        if (exclude == 0)
                        {
                            //removeCount = 0;
                            //for (var j = start; j < i; j++)
                            //{
                            //    if (z[j] != '\n') removeCount++;
                            //    else if (removeCount > 0)
                            //    {
                            //        z = z.Remove(start, removeCount);
                            //        i -= removeCount;
                            //        removeCount = 0;
                            //    }
                            //}
                            removeCount = i - start;
                            var v = z.Substring(start, removeCount);
                            var lfs = v.Length - v.Replace("\n", "").Length;
                            z = z.Remove(start, removeCount);
                            i -= removeCount;
                            z = z.Insert(start, new string('\n', lfs));
                            i += lfs;
                        }
                    }
                    removeCount = 0;
                    for (var j = i; (j < z.Length) && (z[j] != '\n'); j++) removeCount++;
                    z = z.Remove(i, removeCount);
                }
                else if ((z.Substring(i, 6) == "%ifdef" && char.IsWhiteSpace(z[i + 6])) || (z.Substring(i, 7) == "%ifndef" && char.IsWhiteSpace(z[i + 7])))
                {
                    if (exclude > 0)
                        exclude++;
                    else
                    {
                        var j = i + 7;
                        for (; char.IsWhiteSpace(z[j]); j++) { }
                        var n = 0;
                        for (; j + n < z.Length && !char.IsWhiteSpace(z[j + n]); n++) { }
                        exclude = 1;
                        var flag = z.Substring(j, n);
                        for (var k = 0; k < defines.Length; k++)
                            if (defines[k] == flag)
                            {
                                exclude = 0;
                                break;
                            }
                        if (z[i + 3] == 'n') exclude = (exclude == 0 ? 1 : 0);
                        if (exclude > 0)
                        {
                            start = i;
                            start_lineno = lineno;
                        }
                    }
                    removeCount = 0;
                    for (var j = i; (j < z.Length) && (z[j] != '\n'); j++) removeCount++;
                    z = z.Remove(i, removeCount);
                }
            }
            if (exclude > 0)
                throw new Exception(string.Format("unterminated %ifdef starting on line {0}\n", start_lineno));
        }

        public static void Parse(Context ctx)
        {
            var startline = 0;
            var pctx = new ParserContext
            {
                Filename = ctx.Filename,
                Errors = 0,
                FiniteState = FiniteState.INITIALIZE,
            };

            /* Begin by reading the input file */
            if (!File.Exists(pctx.Filename))
            {
                Context.ErrorMsg(ref ctx.Errors, pctx.Filename, 0, "Can't open this file for reading.");
                return;
            }
            var z = File.ReadAllText(pctx.Filename);

            /* Make an initial pass through the file to handle %ifdef and %ifndef */
            PreProcessInput(ref z, new string[] { });

            /* Now scan the text of the input file */
            char c;
            var lineno = 1;
            var nextcp = 0;
            for (var cp = 0; cp < z.Length && (c = z[cp]) != 0; )
            {
                if (c == '\n') lineno++;
                /* Skip all white space */
                if (char.IsWhiteSpace(c)) { cp++; continue; }
                /* Skip C/C++ style comments */
                if (c == '/' && z[cp + 1] == '/')
                {
                    cp += 2;
                    while (cp < z.Length && (c = z[cp]) != 0 && c != '\n') cp++;
                    continue;
                }
                if (c == '/' && z[cp + 1] == '*')
                {
                    cp += 2;
                    while (cp < z.Length && (c = z[cp]) != 0 && (c != '/' || z[cp - 1] != '*'))
                    {
                        if (c == '\n') lineno++;
                        cp++;
                    }
                    if (cp < z.Length) cp++;
                    continue;
                }
                pctx.TokenLineno = lineno;
                pctx.TokenStart = cp;
                /* String literals */
                if (c == '\"')
                {
                    cp++;
                    while (cp < z.Length && (c = z[cp]) != 0 && c != '\"')
                    {
                        if (c == '\n') lineno++;
                        cp++;
                    }
                    if (cp < z.Length)
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, startline, "String starting on this line is not terminated before the end of the file.");
                        nextcp = cp;
                    }
                    else
                        nextcp = cp + 1;
                }
                /* A block of C code */
                else if (c == '{')
                {
                    cp++;
                    for (var level = 1; cp < z.Length && (c = z[cp]) != 0 && (level > 1 || c != '}'); cp++)
                    {
                        if (c == '\n') lineno++;
                        else if (c == '{') level++;
                        else if (c == '}') level--;
                        else if (c == '/' && z[cp + 1] == '*') /* Skip comments */
                        {
                            cp += 2;
                            var prevc = default(char);
                            while (cp < z.Length && (c = z[cp]) != 0 && (c != '/' || prevc != '*'))
                            {
                                if (c == '\n') lineno++;
                                prevc = c;
                                cp++;
                            }
                        }
                        else if (c == '/' && z[cp + 1] == '/') /* Skip C++ style comments too */
                        {
                            cp += 2;
                            while (cp < z.Length && (c = z[cp]) != 0 && c != '\n') cp++;
                            if (cp < z.Length) lineno++;
                        }
                        else if (c == '\'' || c == '\"') /* String a character literals */
                        {
                            var startchar = c;
                            var prevc = default(char);
                            for (cp++; cp < z.Length && (c = z[cp]) != 0 && (c != startchar || prevc == '\\'); cp++)
                            {
                                if (c == '\n') lineno++;
                                prevc = (prevc == '\\' ? default(char) : c);
                            }
                        }
                    }
                    if (cp == z.Length)
                    {
                        Context.ErrorMsg(ref pctx.Errors, pctx.Filename, pctx.TokenLineno, "C code starting on this line is not terminated before the end of the file.");
                        nextcp = cp;
                    }
                    else
                        nextcp = cp + 1;
                }
                /* Identifiers */
                else if (char.IsLetterOrDigit(c))
                {
                    while (cp < z.Length && (c = z[cp]) != 0 && (char.IsLetterOrDigit(c) || (c == '_'))) cp++;
                    nextcp = cp;
                }
                /* The operator "::=" */
                else if (c == ':' && z[cp + 1] == ':' && z[cp + 2] == '=')
                {
                    cp += 3;
                    nextcp = cp;
                }
                else if ((c == '/' || c == '|') && char.IsLetterOrDigit(z[cp + 1]))
                {
                    cp += 2;
                    while (cp < z.Length && (c = z[cp]) != 0 && (char.IsLetterOrDigit(c) || (c == '_'))) cp++;
                    nextcp = cp;
                }
                /* All other (one character) operators */
                else
                {
                    cp++;
                    nextcp = cp;
                }
                pctx.TokenEnd = cp;
                ParseSingleToken(z, ctx, pctx);
                cp = nextcp;
            }
            ctx.Rule = pctx.FirstRule;
            ctx.Errors = pctx.Errors;
        }
    }
}
