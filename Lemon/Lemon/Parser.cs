using System;
using System.IO;

namespace Lemon
{
    public class Parser
    {
        public enum State
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

        public class pstate
        {
            public const int MAXRHS = 100;
            public string filename;
            public int tokenlineno;
            public int errorcnt;
            public int tokenstart;
            public int tokenend;
            public Lemon gp;
            public State state;
            public Symbol fallback;
            public Symbol lhs;
            public string lhsalias;
            public int nrhs;
            public Symbol[] rhs = new Symbol[MAXRHS];
            public string[] alias = new string[MAXRHS];
            public Rule prevrule;
            public string declkeyword;
            public string declargslot;
            public int insertLineMacro;
            public int decllinenoslot;
            public Association declassoc;
            public int preccounter;
            public Rule firstrule;
            public Rule lastrule;
        }

        private static void ParseSingleToken(string z, pstate psp)
        {
            var x = z.Substring(psp.tokenstart, psp.tokenend - psp.tokenstart);
#if true
            Console.WriteLine("{0}:{1}: Token=[{2}] state={3}", psp.filename, psp.tokenlineno, x, psp.state);
#endif
            switch (psp.state)
            {
                case State.INITIALIZE:
                case State.WAITING_FOR_DECL_OR_RULE:
                    if (psp.state == State.INITIALIZE)
                    {
                        psp.prevrule = null;
                        psp.preccounter = 0;
                        psp.firstrule = psp.lastrule = null;
                        psp.gp.nrule = 0;
                    }
                    // WAITING_FOR_DECL_OR_RULE:
                    if (x[0] == '%')
                        psp.state = State.WAITING_FOR_DECL_KEYWORD;
                    else if (char.IsLower(x[0]))
                    {
                        psp.lhs = Symbol.New(x);
                        psp.nrhs = 0;
                        psp.lhsalias = null;
                        psp.state = State.WAITING_FOR_ARROW;
                    }
                    else if (x[0] == '{')
                    {
                        if (psp.prevrule == null)
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "There is no prior rule opon which to attach the code fragment which begins on this line.");
                        else if (psp.prevrule.code != null)
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Code fragment beginning on this line is not the first to follow the previous rule.");
                        else
                        {
                            psp.prevrule.line = psp.tokenlineno;
                            psp.prevrule.code = x.Substring(1);
                        }
                    }
                    else if (x[0] == '[')
                        psp.state = State.PRECEDENCE_MARK_1;
                    else
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Token \"{0}\" should be either \"%\" or a nonterminal name.", x);
                    break;
                case State.PRECEDENCE_MARK_1:
                    if (!char.IsUpper(x[0]))
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "The precedence symbol must be a terminal.");
                    else if (psp.prevrule == null)
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "There is no prior rule to assign precedence \"[{0}]\".", x);
                    else if (psp.prevrule.precsym != null)
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Precedence mark on this line is not the first to follow the previous rule.");
                    else
                        psp.prevrule.precsym = Symbol.New(x);
                    psp.state = State.PRECEDENCE_MARK_2;
                    break;
                case State.PRECEDENCE_MARK_2:
                    if (x[0] != ']')
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Missing \"]\" on precedence mark.");
                    psp.state = State.WAITING_FOR_DECL_OR_RULE;
                    break;
                case State.WAITING_FOR_ARROW:
                    if ((x[0] == ':') && (x[1] == ':') && (x[2] == '='))
                        psp.state = State.IN_RHS;
                    else if (x[0] == '(')
                        psp.state = State.LHS_ALIAS_1;
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Expected to see a \":\" following the LHS symbol \"{0}\".", psp.lhs.Name);
                        psp.state = State.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case State.LHS_ALIAS_1:
                    if (char.IsLetter(x[0]))
                    {
                        psp.lhsalias = x;
                        psp.state = State.LHS_ALIAS_2;
                    }
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "\"{0}\" is not a valid alias for the LHS \"{1}\"\n", x, psp.lhs.Name);
                        psp.state = State.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case State.LHS_ALIAS_2:
                    if (x[0] == ')')
                        psp.state = State.LHS_ALIAS_3;
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Missing \")\" following LHS alias name \"{0}\".", psp.lhsalias);
                        psp.state = State.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case State.LHS_ALIAS_3:
                    if (x[0] == ':' && x[1] == ':' && x[2] == '=')
                        psp.state = State.IN_RHS;
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Missing \".\" following: \"{0}({1})\".", psp.lhs.Name, psp.lhsalias);
                        psp.state = State.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case State.IN_RHS:
                    if (x[0] == '.')
                    {
                        var rp = new Rule
                        {
                            ruleline = psp.tokenlineno,
                            rhs = new Symbol[psp.nrhs],
                            rhsalias = new string[psp.nrhs],
                            lhs = psp.lhs,
                            lhsalias = psp.lhsalias,
                            nrhs = psp.nrhs,
                            index = psp.gp.nrule++,
                        };
                        for (var i = 0; i < psp.nrhs; i++)
                        {
                            rp.rhs[i] = psp.rhs[i];
                            rp.rhsalias[i] = psp.alias[i];
                        }
                        rp.nextlhs = rp.lhs.Rule;
                        rp.lhs.Rule = rp;
                        rp.next = null;
                        if (psp.firstrule == null)
                            psp.firstrule = psp.lastrule = rp;
                        else
                        {
                            psp.lastrule.next = rp;
                            psp.lastrule = rp;
                        }
                        psp.prevrule = rp;
                        psp.state = State.WAITING_FOR_DECL_OR_RULE;
                    }
                    else if (char.IsLetter(x[0]))
                    {
                        if (psp.nrhs >= pstate.MAXRHS)
                        {
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Too many symbols on RHS of rule beginning at \"%s\".", x);
                            psp.state = State.RESYNC_AFTER_RULE_ERROR;
                        }
                        else
                        {
                            psp.rhs[psp.nrhs] = Symbol.New(x);
                            psp.alias[psp.nrhs] = null;
                            psp.nrhs++;
                        }
                    }
                    else if (((x[0] == '|') || (x[0] == '/')) && (psp.nrhs > 0))
                    {
                        var msp = psp.rhs[psp.nrhs - 1];
                        if (msp.Type != SymbolType.MultiTerminal)
                        {
                            var origsp = msp;
                            msp = new Symbol
                            {
                                Type = SymbolType.MultiTerminal,
                                subsym = new[] { origsp },
                                Name = origsp.Name,
                            };
                            psp.rhs[psp.nrhs - 1] = msp;
                        }
                        Array.Resize(ref msp.subsym, msp.subsym.Length + 1);
                        msp.subsym[msp.subsym.Length] = Symbol.New(x.Substring(1));
                        if (char.IsLower(x[1]) || char.IsLower(msp.subsym[0].Name[0]))
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Cannot form a compound containing a non-terminal");
                    }
                    else if (x[0] == '(' && psp.nrhs > 0)
                        psp.state = State.RHS_ALIAS_1;
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Illegal character on RHS of rule: \"{0}\".", x);
                        psp.state = State.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case State.RHS_ALIAS_1:
                    if (char.IsLetter(x[0]))
                    {
                        psp.alias[psp.nrhs - 1] = x;
                        psp.state = State.RHS_ALIAS_2;
                    }
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "\"{0}\" is not a valid alias for the RHS symbol \"{1}\"\n", x, psp.rhs[psp.nrhs - 1].Name);
                        psp.state = State.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case State.RHS_ALIAS_2:
                    if (x[0] == ')')
                        psp.state = State.IN_RHS;
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Missing \")\" following LHS alias name \"{0}\".", psp.lhsalias);
                        psp.state = State.RESYNC_AFTER_RULE_ERROR;
                    }
                    break;
                case State.WAITING_FOR_DECL_KEYWORD:
                    if (char.IsLetter(x[0]))
                    {
                        psp.declkeyword = x;
                        psp.declargslot = null;
                        psp.decllinenoslot = 0;
                        psp.insertLineMacro = 1;
                        psp.state = State.WAITING_FOR_DECL_ARG;
                        if (x == "name")
                        {
                            psp.declargslot = psp.gp.name;
                            psp.insertLineMacro = 0;
                        }
                        else if (x == "include")
                            psp.declargslot = psp.gp.include;
                        else if (x == "code")
                            psp.declargslot = psp.gp.extracode;
                        else if (x == "token_destructor")
                            psp.declargslot = psp.gp.tokendest;
                        else if (x == "default_destructor")
                            psp.declargslot = psp.gp.vardest;
                        else if (x == "token_prefix")
                        {
                            psp.declargslot = psp.gp.tokenprefix;
                            psp.insertLineMacro = 0;
                        }
                        else if (x == "syntax_error")
                            psp.declargslot = psp.gp.error;
                        else if (x == "parse_accept")
                            psp.declargslot = psp.gp.accept;
                        else if (x == "parse_failure")
                            psp.declargslot = psp.gp.failure;
                        else if (x == "stack_overflow")
                            psp.declargslot = psp.gp.overflow;
                        else if (x == "extra_argument")
                        {
                            psp.declargslot = psp.gp.arg;
                            psp.insertLineMacro = 0;
                        }
                        else if (x == "token_type")
                        {
                            psp.declargslot = psp.gp.tokentype;
                            psp.insertLineMacro = 0;
                        }
                        else if (x == "default_type")
                        {
                            psp.declargslot = psp.gp.vartype;
                            psp.insertLineMacro = 0;
                        }
                        else if (x == "stack_size")
                        {
                            psp.declargslot = psp.gp.stacksize;
                            psp.insertLineMacro = 0;
                        }
                        else if (x == "start_symbol")
                        {
                            psp.declargslot = psp.gp.start;
                            psp.insertLineMacro = 0;
                        }
                        else if (x == "left")
                        {
                            psp.preccounter++;
                            psp.declassoc = Association.Left;
                            psp.state = State.WAITING_FOR_PRECEDENCE_SYMBOL;
                        }
                        else if (x == "right")
                        {
                            psp.preccounter++;
                            psp.declassoc = Association.Right;
                            psp.state = State.WAITING_FOR_PRECEDENCE_SYMBOL;
                        }
                        else if (x == "nonassoc")
                        {
                            psp.preccounter++;
                            psp.declassoc = Association.None;
                            psp.state = State.WAITING_FOR_PRECEDENCE_SYMBOL;
                        }
                        else if (x == "destructor")
                            psp.state = State.WAITING_FOR_DESTRUCTOR_SYMBOL;
                        else if (x == "type")
                            psp.state = State.WAITING_FOR_DATATYPE_SYMBOL;
                        else if (x == "fallback")
                        {
                            psp.fallback = null;
                            psp.state = State.WAITING_FOR_FALLBACK_ID;
                        }
                        else if (x == "wildcard")
                            psp.state = State.WAITING_FOR_WILDCARD_ID;
                        else
                        {
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Unknown declaration keyword: \"%{0}\".", x);
                            psp.state = State.RESYNC_AFTER_DECL_ERROR;
                        }
                    }
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Illegal declaration keyword: \"{0}\".", x);
                        psp.state = State.RESYNC_AFTER_DECL_ERROR;
                    }
                    break;
                case State.WAITING_FOR_DESTRUCTOR_SYMBOL:
                    if (!char.IsLetter(x[0]))
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Symbol name missing after %destructor keyword");
                        psp.state = State.RESYNC_AFTER_DECL_ERROR;
                    }
                    else
                    {
                        var sp = Symbol.New(x);
                        psp.declargslot = sp.destructor;
                        psp.decllinenoslot = sp.destLineno;
                        psp.insertLineMacro = 1;
                        psp.state = State.WAITING_FOR_DECL_ARG;
                    }
                    break;
                case State.WAITING_FOR_DATATYPE_SYMBOL:
                    if (!char.IsLetter(x[0]))
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Symbol name missing after %type keyword");
                        psp.state = State.RESYNC_AFTER_DECL_ERROR;
                    }
                    else
                    {
                        var sp = Symbol.Symbols[x];
                        if ((sp != null) && (sp.datatype != null))
                        {
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Symbol %type \"{0}\" already defined", x);
                            psp.state = State.RESYNC_AFTER_DECL_ERROR;
                        }
                        else
                        {
                            if (sp == null)
                                sp = Symbol.New(x);
                            psp.declargslot = sp.datatype;
                            psp.insertLineMacro = 0;
                            psp.state = State.WAITING_FOR_DECL_ARG;
                        }
                    }
                    break;
                case State.WAITING_FOR_PRECEDENCE_SYMBOL:
                    if (x[0] == '.')
                        psp.state = State.WAITING_FOR_DECL_OR_RULE;
                    else if (char.IsLetter(x[0]))
                    {
                        var sp = Symbol.New(x);
                        if (sp.Precedence >= 0)
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Symbol \"{0}\" has already be given a precedence.", x);
                        else
                        {
                            sp.Precedence = psp.preccounter;
                            sp.Association = psp.declassoc;
                        }
                    }
                    else
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Can't assign a precedence to \"{0}\".", x);
                    break;
                case State.WAITING_FOR_DECL_ARG:
                    if ((x[0] == '{') || (x[0] == '\"') || char.IsLetterOrDigit(x[0]))
                    {
                        //const char *zOld, *zNew;
                        //char *zBuf, *z;
                        //int nOld, n, nLine, nNew, nBack;
                        //int addLineMacro;
                        //char zLine[50];
                        //zNew = x;
                        //if( zNew[0]=='"' || zNew[0]=='{' ) zNew++;
                        //nNew = lemonStrlen(zNew);
                        //if( *psp.declargslot ){
                        //  zOld = *psp.declargslot;
                        //}else{
                        //  zOld = "";
                        //}
                        //nOld = lemonStrlen(zOld);
                        //n = nOld + nNew + 20;
                        //addLineMacro = !psp.gp.nolinenosflag && psp.insertLineMacro && (psp.decllinenoslot==0 || psp.decllinenoslot[0]!=0);
                        //if( addLineMacro ){
                        //  for(z=psp.filename, nBack=0; *z; z++){
                        //    if( *z=='\\' ) nBack++;
                        //  }
                        //  sprintf(zLine, "#line %d ", psp.tokenlineno);
                        //  nLine = lemonStrlen(zLine);
                        //  n += nLine + lemonStrlen(psp.filename) + nBack;
                        //}
                        //*psp.declargslot = (char *) realloc(*psp.declargslot, n);
                        //zBuf = *psp.declargslot + nOld;
                        //if( addLineMacro ){
                        //  if( nOld && zBuf[-1]!='\n' ){
                        //    *(zBuf++) = '\n';
                        //  }
                        //  memcpy(zBuf, zLine, nLine);
                        //  zBuf += nLine;
                        //  *(zBuf++) = '"';
                        //  for(z=psp.filename; *z; z++){
                        //    if( *z=='\\' ){
                        //      *(zBuf++) = '\\';
                        //    }
                        //    *(zBuf++) = *z;
                        //  }
                        //  *(zBuf++) = '"';
                        //  *(zBuf++) = '\n';
                        //}
                        //if( psp.decllinenoslot && psp.decllinenoslot[0]==0 ){
                        //  psp.decllinenoslot[0] = psp.tokenlineno;
                        //}
                        //memcpy(zBuf, zNew, nNew);
                        //zBuf += nNew;
                        //*zBuf = 0;
                        //psp.state = WAITING_FOR_DECL_OR_RULE;
                    }
                    else
                    {
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Illegal argument to %{0}: {1}", psp.declkeyword, x);
                        psp.state = State.RESYNC_AFTER_DECL_ERROR;
                    }
                    break;
                case State.WAITING_FOR_FALLBACK_ID:
                    if (x[0] == '.')
                        psp.state = State.WAITING_FOR_DECL_OR_RULE;
                    else if (!char.IsUpper(x[0]))
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "%fallback argument \"{0}\" should be a token", x);
                    else
                    {
                        var sp = Symbol.New(x);
                        if (psp.fallback == null)
                            psp.fallback = sp;
                        else if (sp.Fallback != null)
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "More than one fallback assigned to token {0}", x);
                        else
                        {
                            sp.Fallback = psp.fallback;
                            psp.gp.has_fallback = 1;
                        }
                    }
                    break;
                case State.WAITING_FOR_WILDCARD_ID:
                    if (x[0] == '.')
                        psp.state = State.WAITING_FOR_DECL_OR_RULE;
                    else if (!char.IsUpper(x[0]))
                        Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "%wildcard argument \"{0}\" should be a token", x);
                    else
                    {
                        var sp = Symbol.New(x);
                        if (psp.gp.wildcard == null)
                            psp.gp.wildcard = sp;
                        else
                            Lemon.ErrorMsg(ref psp.errorcnt, psp.filename, psp.tokenlineno, "Extra wildcard to token: {0}", x);
                    }
                    break;
                case State.RESYNC_AFTER_RULE_ERROR:
                case State.RESYNC_AFTER_DECL_ERROR:
                    if (x[0] == '.')
                        psp.state = State.WAITING_FOR_DECL_OR_RULE;
                    if (x[0] == '%')
                        psp.state = State.WAITING_FOR_DECL_KEYWORD;
                    break;
            }
        }

        static void PreProcessInput(string z, string[] defines)
        {
            var exclude = 0;
            var start = 0;
            var lineno = 1;
            var start_lineno = 1;
            for (var i = 0; i < z.Length; i++)
            {
                if (z[i] == '\n')
                    lineno++;
                if ((z[i] != '%') || ((i > 0) && (z[i - 1] != '\n')))
                    continue;
                if ((z.Substring(i, 6) == "%endif") && char.IsWhiteSpace(z[i + 6]))
                {
                    if (exclude > 0)
                    {
                        exclude--;
                        if (exclude == 0)
                            for (var j = start; j < i; j++)
                                if (z[j] != '\n')
                                    ; //z[j] = ' ';
                    }
                    for (var j = i; (j < z.Length) && (z[j] != '\n'); j++)
                        ; //z[j] = ' ';
                }
                else if (((z.Substring(i, 6) == "%ifdef") && char.IsWhiteSpace(z[i + 6])) || ((z.Substring(i, 7) == "%ifndef") && char.IsWhiteSpace(z[i + 7])))
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
                        if (defines != null)
                            for (var k = 0; k < defines.Length; k++)
                                if (defines[k] == z.Substring(j, n))
                                {
                                    exclude = 0;
                                    break;
                                }
                        if (z[i + 3] == 'n')
                            exclude = (exclude == 0 ? 1 : 0);
                        if (exclude > 0)
                        {
                            start = i;
                            start_lineno = lineno;
                        }
                    }
                    for (var j = i; (j < z.Length) && (z[j] != '\n'); j++)
                        ; // z[j] = ' ';
                }
            }
            if (exclude > 0)
                throw new Exception(string.Format("unterminated %ifdef starting on line {0}\n", start_lineno));
        }

        private static bool NextZ(string s, int i, out char c)
        {
            if (i < s.Length)
            {
                c = default(char);
                return false;
            }
            c = s[i];
            return true;
        }

        public void Parse(Lemon gp)
        {
            int startline = 0;
            var ps = new pstate
            {
                gp = gp,
                filename = gp.filename,
                errorcnt = 0,
                state = State.INITIALIZE,
            };

            /* Begin by reading the input file */
            if (!File.Exists(ps.filename))
            {
                Lemon.ErrorMsg(ref gp.errorcnt, ps.filename, 0, "Can't open this file for reading.");
                return;
            }
            var z = File.ReadAllText(ps.filename);

            /* Make an initial pass through the file to handle %ifdef and %ifndef */
            PreProcessInput(z, null);

            /* Now scan the text of the input file */
            var lineno = 1;
            char c;
            var nextcp = 0;
            for (var cp = 0; NextZ(z, cp, out c); )
            {
                if (c == '\n')
                    lineno++;
                /* Skip all white space */
                if (char.IsWhiteSpace(c))
                {
                    cp++;
                    continue;
                }
                /* Skip C/C++ style comments */
                if ((c == '/') && (z[cp + 1] == '/'))
                {
                    cp += 2;
                    while (NextZ(z, cp, out c) && (c != '\n'))
                        cp++;
                    continue;
                }
                if ((c == '/') && (z[cp + 1] == '*'))
                {
                    cp += 2;
                    while (NextZ(z, cp, out c) && ((c != '/') || (z[cp - 1] != '*')))
                    {
                        if (c == '\n')
                            lineno++;
                        cp++;
                    }
                    if (c != default(char))
                        cp++;
                    continue;
                }
                ps.tokenlineno = lineno;
                ps.tokenstart = cp;
                /* String literals */
                if (c == '\"')
                {
                    cp++;
                    while (NextZ(z, cp, out c) && (c != '\"'))
                    {
                        if (c == '\n')
                            lineno++;
                        cp++;
                    }
                    if (c == default(char))
                    {
                        Lemon.ErrorMsg(ref ps.errorcnt, ps.filename, startline, "String starting on this line is not terminated before the end of the file.");
                        nextcp = cp;
                    }
                    else
                        nextcp = cp + 1;
                }
                /* A block of C code */
                else if (c == '{')
                {
                    cp++;
                    for (var level = 1; NextZ(z, cp, out c) && ((level > 1) || (c != '}')); cp++)
                    {
                        if (c == '\n')
                            lineno++;
                        else if (c == '{')
                            level++;
                        else if (c == '}')
                            level--;
                        /* Skip C/C++ comments */
                        else if ((c == '/') && (z[cp + 1] == '/'))
                        {
                            cp += 2;
                            while (NextZ(z, cp, out c) && (c != '\n'))
                                cp++;
                            if (c != default(char))
                                lineno++;
                        }
                        else if ((c == '/') && (z[cp + 1] == '*'))
                        {
                            cp += 2;
                            while (NextZ(z, cp, out c) && ((c != '/') || (z[cp - 1] != '*')))
                            {
                                if (c == '\n')
                                    lineno++;
                                cp++;
                            }
                            //if (c != default(char))
                            //    cp++;
                        }
                        /* String a character literals */
                        else if ((c == '\'') || (c == '\"'))
                        {
                            var startchar = c;
                            var prevc = default(char);
                            for (cp++; NextZ(z, cp, out c) && ((c != startchar) || (prevc == '\\')); cp++)
                            {
                                if (c == '\n')
                                    lineno++;
                                prevc = (prevc == '\\' ? default(char) : c);
                            }
                        }
                    }
                    if (c == default(char))
                    {
                        Lemon.ErrorMsg(ref ps.errorcnt, ps.filename, ps.tokenlineno, "C code starting on this line is not terminated before the end of the file.");
                        nextcp = cp;
                    }
                    else
                        nextcp = cp + 1;
                }
                /* Identifiers */
                else if (char.IsLetter(c))
                {
                    while (NextZ(z, cp, out c) && (char.IsLetter(c) || (c == '_')))
                        cp++;
                    nextcp = cp;
                }
                /* The operator "::=" */
                else if ((c == ':') && (z[cp + 1] == ':') && (z[cp + 2] == '='))
                {
                    cp += 3;
                    nextcp = cp;
                }
                else if (((c == '/') || (c == '|')) && char.IsLetter(z[cp + 1]))
                {
                    cp += 2;
                    while (NextZ(z, cp, out c) && (char.IsLetter(c) || (c == '_')))
                        cp++;
                    nextcp = cp;
                }
                /* All other (one character) operators */
                else
                {
                    cp++;
                    nextcp = cp;
                }
                ps.tokenend = cp;
                ParseSingleToken(z, ps);
                cp = nextcp;
            }
            gp.rule = ps.firstrule;
            gp.errorcnt = ps.errorcnt;
        }
    }
}
