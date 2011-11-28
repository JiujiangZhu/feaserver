using System;
using System.IO;

namespace Lemon
{
    public class Emitter
    {
        
** The following routine emits code for the destructor for the
** symbol sp
*/
void emit_destructor_code(
  FILE *out,
  struct symbol *sp,
  struct lemon *lemp,
  int *lineno
){
 char *cp = 0;

 if( sp->type==TERMINAL ){
   cp = lemp->tokendest;
   if( cp==0 ) return;
   fprintf(out,"{\n"); (*lineno)++;
 }else if( sp->destructor ){
   cp = sp->destructor;
   fprintf(out,"{\n"); (*lineno)++;
   if (!lemp->nolinenosflag) { (*lineno)++; tplt_linedir(out,sp->destLineno,lemp->filename); }
 }else if( lemp->vardest ){
   cp = lemp->vardest;
   if( cp==0 ) return;
   fprintf(out,"{\n"); (*lineno)++;
 }else{
   assert( 0 );  /* Cannot happen */
 }
 for(; *cp; cp++){
   if( *cp=='$' && cp[1]=='$' ){
     fprintf(out,"(yypminor->yy%d)",sp->dtnum);
     cp++;
     continue;
   }
   if( *cp=='\n' ) (*lineno)++;
   fputc(*cp,out);
 }
 fprintf(out,"\n"); (*lineno)++;
 if (!lemp->nolinenosflag) { 
   (*lineno)++; tplt_linedir(out,*lineno,lemp->outname); 
 }
 fprintf(out,"}\n"); (*lineno)++;
 return;
}

/*
** Return TRUE (non-zero) if the given symbol has a destructor.
*/
int has_destructor(struct symbol *sp, struct lemon *lemp)
{
  int ret;
  if( sp->type==TERMINAL ){
    ret = lemp->tokendest!=0;
  }else{
    ret = lemp->vardest!=0 || sp->destructor!=0;
  }
  return ret;
}

/*
** Append text to a dynamically allocated string.  If zText is 0 then
** reset the string to be empty again.  Always return the complete text
** of the string (which is overwritten with each call).
**
** n bytes of zText are stored.  If n==0 then all of zText up to the first
** \000 terminator is stored.  zText can contain up to two instances of
** %d.  The values of p1 and p2 are written into the first and second
** %d.
**
** If n==-1, then the previous character is overwritten.
*/
PRIVATE char *append_str(const char *zText, int n, int p1, int p2){
  static char empty[1] = { 0 };
  static char *z = 0;
  static int alloced = 0;
  static int used = 0;
  int c;
  char zInt[40];
  if( zText==0 ){
    used = 0;
    return z;
  }
  if( n<=0 ){
    if( n<0 ){
      used += n;
      assert( used>=0 );
    }
    n = lemonStrlen(zText);
  }
  if( (int) (n+sizeof(zInt)*2+used) >= alloced ){
    alloced = n + sizeof(zInt)*2 + used + 200;
    z = (char *) realloc(z,  alloced);
  }
  if( z==0 ) return empty;
  while( n-- > 0 ){
    c = *(zText++);
    if( c=='%' && n>0 && zText[0]=='d' ){
      sprintf(zInt, "%d", p1);
      p1 = p2;
      strcpy(&z[used], zInt);
      used += lemonStrlen(&z[used]);
      zText++;
      n--;
    }else{
      z[used++] = c;
    }
  }
  z[used] = 0;
  return z;
}

/*
** zCode is a string that is the action associated with a rule.  Expand
** the symbols in this string so that the refer to elements of the parser
** stack.
*/
PRIVATE void translate_code(struct lemon *lemp, struct rule *rp){
  char *cp, *xp;
  int i;
  char lhsused = 0;    /* True if the LHS element has been used */
  char used[MAXRHS];   /* True for each RHS element which is used */

  for(i=0; i<rp->nrhs; i++) used[i] = 0;
  lhsused = 0;

  if( rp->code==0 ){
    static char newlinestr[2] = { '\n', '\0' };
    rp->code = newlinestr;
    rp->line = rp->ruleline;
  }

  append_str(0,0,0,0);

  /* This const cast is wrong but harmless, if we're careful. */
  for(cp=(char *)rp->code; *cp; cp++){
    if( isalpha(*cp) && (cp==rp->code || (!isalnum(cp[-1]) && cp[-1]!='_')) ){
      char saved;
      for(xp= &cp[1]; isalnum(*xp) || *xp=='_'; xp++);
      saved = *xp;
      *xp = 0;
      if( rp->lhsalias && strcmp(cp,rp->lhsalias)==0 ){
        append_str("yygotominor.yy%d",0,rp->lhs->dtnum,0);
        cp = xp;
        lhsused = 1;
      }else{
        for(i=0; i<rp->nrhs; i++){
          if( rp->rhsalias[i] && strcmp(cp,rp->rhsalias[i])==0 ){
            if( cp!=rp->code && cp[-1]=='@' ){
              /* If the argument is of the form @X then substituted
              ** the token number of X, not the value of X */
              append_str("yymsp[%d].major",-1,i-rp->nrhs+1,0);
            }else{
              struct symbol *sp = rp->rhs[i];
              int dtnum;
              if( sp->type==MULTITERMINAL ){
                dtnum = sp->subsym[0]->dtnum;
              }else{
                dtnum = sp->dtnum;
              }
              append_str("yymsp[%d].minor.yy%d",0,i-rp->nrhs+1, dtnum);
            }
            cp = xp;
            used[i] = 1;
            break;
          }
        }
      }
      *xp = saved;
    }
    append_str(cp, 1, 0, 0);
  } /* End loop */

  /* Check to make sure the LHS has been used */
  if( rp->lhsalias && !lhsused ){
    ErrorMsg(lemp->filename,rp->ruleline,
      "Label \"%s\" for \"%s(%s)\" is never used.",
        rp->lhsalias,rp->lhs->name,rp->lhsalias);
    lemp->errorcnt++;
  }

  /* Generate destructor code for RHS symbols which are not used in the
  ** reduce code */
  for(i=0; i<rp->nrhs; i++){
    if( rp->rhsalias[i] && !used[i] ){
      ErrorMsg(lemp->filename,rp->ruleline,
        "Label %s for \"%s(%s)\" is never used.",
        rp->rhsalias[i],rp->rhs[i]->name,rp->rhsalias[i]);
      lemp->errorcnt++;
    }else if( rp->rhsalias[i]==0 ){
      if( has_destructor(rp->rhs[i],lemp) ){
        append_str("  yy_destructor(yypParser,%d,&yymsp[%d].minor);\n", 0,
           rp->rhs[i]->index,i-rp->nrhs+1);
      }else{
        /* No destructor defined for this term */
      }
    }
  }
  if( rp->code ){
    cp = append_str(0,0,0,0);
    rp->code = Strsafe(cp?cp:"");
  }
}

/* 
** Generate code which executes when the rule "rp" is reduced.  Write
** the code to "out".  Make sure lineno stays up-to-date.
*/
PRIVATE void emit_code(
  FILE *out,
  struct rule *rp,
  struct lemon *lemp,
  int *lineno
){
 const char *cp;

 /* Generate code to do the reduce action */
 if( rp->code ){
   if (!lemp->nolinenosflag) { (*lineno)++; tplt_linedir(out,rp->line,lemp->filename); }
   fprintf(out,"{%s",rp->code);
   for(cp=rp->code; *cp; cp++){
     if( *cp=='\n' ) (*lineno)++;
   } /* End loop */
   fprintf(out,"}\n"); (*lineno)++;
   if (!lemp->nolinenosflag) { (*lineno)++; tplt_linedir(out,*lineno,lemp->outname); }
 } /* End if( rp->code ) */

 return;
}

/*
** Print the definition of the union used for the parser's data stack.
** This union contains fields for every possible data type for tokens
** and nonterminals.  In the process of computing and printing this
** union, also set the ".dtnum" field of every terminal and nonterminal
** symbol.
*/
void print_stack_union(
  FILE *out,                  /* The output stream */
  struct lemon *lemp,         /* The main info structure for this parser */
  int *plineno,               /* Pointer to the line number */
  int mhflag                  /* True if generating makeheaders output */
){
  int lineno = *plineno;    /* The line number of the output */
  char **types;             /* A hash table of datatypes */
  int arraysize;            /* Size of the "types" array */
  int maxdtlength;          /* Maximum length of any ".datatype" field. */
  char *stddt;              /* Standardized name for a datatype */
  int i,j;                  /* Loop counters */
  int hash;                 /* For hashing the name of a type */
  const char *name;         /* Name of the parser */

  /* Allocate and initialize types[] and allocate stddt[] */
  arraysize = lemp->nsymbol * 2;
  types = (char**)calloc( arraysize, sizeof(char*) );
  if( types==0 ){
    fprintf(stderr,"Out of memory.\n");
    exit(1);
  }
  for(i=0; i<arraysize; i++) types[i] = 0;
  maxdtlength = 0;
  if( lemp->vartype ){
    maxdtlength = lemonStrlen(lemp->vartype);
  }
  for(i=0; i<lemp->nsymbol; i++){
    int len;
    struct symbol *sp = lemp->symbols[i];
    if( sp->datatype==0 ) continue;
    len = lemonStrlen(sp->datatype);
    if( len>maxdtlength ) maxdtlength = len;
  }
  stddt = (char*)malloc( maxdtlength*2 + 1 );
  if( stddt==0 ){
    fprintf(stderr,"Out of memory.\n");
    exit(1);
  }

  /* Build a hash table of datatypes. The ".dtnum" field of each symbol
  ** is filled in with the hash index plus 1.  A ".dtnum" value of 0 is
  ** used for terminal symbols.  If there is no %default_type defined then
  ** 0 is also used as the .dtnum value for nonterminals which do not specify
  ** a datatype using the %type directive.
  */
  for(i=0; i<lemp->nsymbol; i++){
    struct symbol *sp = lemp->symbols[i];
    char *cp;
    if( sp==lemp->errsym ){
      sp->dtnum = arraysize+1;
      continue;
    }
    if( sp->type!=NONTERMINAL || (sp->datatype==0 && lemp->vartype==0) ){
      sp->dtnum = 0;
      continue;
    }
    cp = sp->datatype;
    if( cp==0 ) cp = lemp->vartype;
    j = 0;
    while( isspace(*cp) ) cp++;
    while( *cp ) stddt[j++] = *cp++;
    while( j>0 && isspace(stddt[j-1]) ) j--;
    stddt[j] = 0;
    if( lemp->tokentype && strcmp(stddt, lemp->tokentype)==0 ){
      sp->dtnum = 0;
      continue;
    }
    hash = 0;
    for(j=0; stddt[j]; j++){
      hash = hash*53 + stddt[j];
    }
    hash = (hash & 0x7fffffff)%arraysize;
    while( types[hash] ){
      if( strcmp(types[hash],stddt)==0 ){
        sp->dtnum = hash + 1;
        break;
      }
      hash++;
      if( hash>=arraysize ) hash = 0;
    }
    if( types[hash]==0 ){
      sp->dtnum = hash + 1;
      types[hash] = (char*)malloc( lemonStrlen(stddt)+1 );
      if( types[hash]==0 ){
        fprintf(stderr,"Out of memory.\n");
        exit(1);
      }
      strcpy(types[hash],stddt);
    }
  }

  /* Print out the definition of YYTOKENTYPE and YYMINORTYPE */
  name = lemp->name ? lemp->name : "Parse";
  lineno = *plineno;
  if( mhflag ){ fprintf(out,"#if INTERFACE\n"); lineno++; }
  fprintf(out,"#define %sTOKENTYPE %s\n",name,
    lemp->tokentype?lemp->tokentype:"void*");  lineno++;
  if( mhflag ){ fprintf(out,"#endif\n"); lineno++; }
  fprintf(out,"typedef union {\n"); lineno++;
  fprintf(out,"  int yyinit;\n"); lineno++;
  fprintf(out,"  %sTOKENTYPE yy0;\n",name); lineno++;
  for(i=0; i<arraysize; i++){
    if( types[i]==0 ) continue;
    fprintf(out,"  %s yy%d;\n",types[i],i+1); lineno++;
    free(types[i]);
  }
  if( lemp->errsym->useCnt ){
    fprintf(out,"  int yy%d;\n",lemp->errsym->dtnum); lineno++;
  }
  free(stddt);
  free(types);
  fprintf(out,"} YYMINORTYPE;\n"); lineno++;
  *plineno = lineno;
}

/*
** Return the name of a C datatype able to represent values between
** lwr and upr, inclusive.
*/
static const char *minimum_size_type(int lwr, int upr){
  if( lwr>=0 ){
    if( upr<=255 ){
      return "unsigned char";
    }else if( upr<65535 ){
      return "unsigned short int";
    }else{
      return "unsigned int";
    }
  }else if( lwr>=-127 && upr<=127 ){
    return "signed char";
  }else if( lwr>=-32767 && upr<32767 ){
    return "short";
  }else{
    return "int";
  }
}

/*
** Each state contains a set of token transaction and a set of
** nonterminal transactions.  Each of these sets makes an instance
** of the following structure.  An array of these structures is used
** to order the creation of entries in the yy_action[] table.
*/
struct axset {
  struct state *stp;   /* A pointer to a state */
  int isTkn;           /* True to use tokens.  False for non-terminals */
  int nAction;         /* Number of actions */
  int iOrder;          /* Original order of action sets */
};

/*
** Compare to axset structures for sorting purposes
*/
static int axset_compare(const void *a, const void *b){
  struct axset *p1 = (struct axset*)a;
  struct axset *p2 = (struct axset*)b;
  int c;
  c = p2->nAction - p1->nAction;
  if( c==0 ){
    c = p2->iOrder - p1->iOrder;
  }
  assert( c!=0 || p1==p2 );
  return c;
}

/*
** Write text on "out" that describes the rule "rp".
*/
static void writeRuleText(FILE *out, struct rule *rp){
  int j;
  fprintf(out,"%s ::=", rp->lhs->name);
  for(j=0; j<rp->nrhs; j++){
    struct symbol *sp = rp->rhs[j];
    fprintf(out," %s", sp->name);
    if( sp->type==MULTITERMINAL ){
      int k;
      for(k=1; k<sp->nsubsym; k++){
        fprintf(out,"|%s",sp->subsym[k]->name);
      }
    }
  }
}


/* Generate C source code for the parser */
void ReportTable(
  struct lemon *lemp,
  int mhflag     /* Output in makeheaders format if true */
){
  FILE *out, *in;
  char line[LINESIZE];
  int  lineno;
  struct state *stp;
  struct action *ap;
  struct rule *rp;
  struct acttab *pActtab;
  int i, j, n;
  const char *name;
  int mnTknOfst, mxTknOfst;
  int mnNtOfst, mxNtOfst;
  struct axset *ax;

  in = tplt_open(lemp);
  if( in==0 ) return;
  out = file_open(lemp,".c","wb");
  if( out==0 ){
    fclose(in);
    return;
  }
  lineno = 1;
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate the include code, if any */
  tplt_print(out,lemp,lemp->include,&lineno);
  if( mhflag ){
    char *name = file_makename(lemp, ".h");
    fprintf(out,"#include \"%s\"\n", name); lineno++;
    free(name);
  }
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate #defines for all tokens */
  if( mhflag ){
    const char *prefix;
    fprintf(out,"#if INTERFACE\n"); lineno++;
    if( lemp->tokenprefix ) prefix = lemp->tokenprefix;
    else                    prefix = "";
    for(i=1; i<lemp->nterminal; i++){
      fprintf(out,"#define %s%-30s %2d\n",prefix,lemp->symbols[i]->name,i);
      lineno++;
    }
    fprintf(out,"#endif\n"); lineno++;
  }
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate the defines */
  fprintf(out,"#define YYCODETYPE %s\n",
    minimum_size_type(0, lemp->nsymbol+1)); lineno++;
  fprintf(out,"#define YYNOCODE %d\n",lemp->nsymbol+1);  lineno++;
  fprintf(out,"#define YYACTIONTYPE %s\n",
    minimum_size_type(0, lemp->nstate+lemp->nrule+5));  lineno++;
  if( lemp->wildcard ){
    fprintf(out,"#define YYWILDCARD %d\n",
       lemp->wildcard->index); lineno++;
  }
  print_stack_union(out,lemp,&lineno,mhflag);
  fprintf(out, "#ifndef YYSTACKDEPTH\n"); lineno++;
  if( lemp->stacksize ){
    fprintf(out,"#define YYSTACKDEPTH %s\n",lemp->stacksize);  lineno++;
  }else{
    fprintf(out,"#define YYSTACKDEPTH 100\n");  lineno++;
  }
  fprintf(out, "#endif\n"); lineno++;
  if( mhflag ){
    fprintf(out,"#if INTERFACE\n"); lineno++;
  }
  name = lemp->name ? lemp->name : "Parse";
  if( lemp->arg && lemp->arg[0] ){
    int i;
    i = lemonStrlen(lemp->arg);
    while( i>=1 && isspace(lemp->arg[i-1]) ) i--;
    while( i>=1 && (isalnum(lemp->arg[i-1]) || lemp->arg[i-1]=='_') ) i--;
    fprintf(out,"#define %sARG_SDECL %s;\n",name,lemp->arg);  lineno++;
    fprintf(out,"#define %sARG_PDECL ,%s\n",name,lemp->arg);  lineno++;
    fprintf(out,"#define %sARG_FETCH %s = yypParser->%s\n",
                 name,lemp->arg,&lemp->arg[i]);  lineno++;
    fprintf(out,"#define %sARG_STORE yypParser->%s = %s\n",
                 name,&lemp->arg[i],&lemp->arg[i]);  lineno++;
  }else{
    fprintf(out,"#define %sARG_SDECL\n",name);  lineno++;
    fprintf(out,"#define %sARG_PDECL\n",name);  lineno++;
    fprintf(out,"#define %sARG_FETCH\n",name); lineno++;
    fprintf(out,"#define %sARG_STORE\n",name); lineno++;
  }
  if( mhflag ){
    fprintf(out,"#endif\n"); lineno++;
  }
  fprintf(out,"#define YYNSTATE %d\n",lemp->nstate);  lineno++;
  fprintf(out,"#define YYNRULE %d\n",lemp->nrule);  lineno++;
  if( lemp->errsym->useCnt ){
    fprintf(out,"#define YYERRORSYMBOL %d\n",lemp->errsym->index);  lineno++;
    fprintf(out,"#define YYERRSYMDT yy%d\n",lemp->errsym->dtnum);  lineno++;
  }
  if( lemp->has_fallback ){
    fprintf(out,"#define YYFALLBACK 1\n");  lineno++;
  }
  tplt_xfer(lemp->name,in,out,&lineno);

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
  ax = (struct axset *) calloc(lemp->nstate*2, sizeof(ax[0]));
  if( ax==0 ){
    fprintf(stderr,"malloc failed\n");
    exit(1);
  }
  for(i=0; i<lemp->nstate; i++){
    stp = lemp->sorted[i];
    ax[i*2].stp = stp;
    ax[i*2].isTkn = 1;
    ax[i*2].nAction = stp->nTknAct;
    ax[i*2+1].stp = stp;
    ax[i*2+1].isTkn = 0;
    ax[i*2+1].nAction = stp->nNtAct;
  }
  mxTknOfst = mnTknOfst = 0;
  mxNtOfst = mnNtOfst = 0;

  /* Compute the action table.  In order to try to keep the size of the
  ** action table to a minimum, the heuristic of placing the largest action
  ** sets first is used.
  */
  for(i=0; i<lemp->nstate*2; i++) ax[i].iOrder = i;
  qsort(ax, lemp->nstate*2, sizeof(ax[0]), axset_compare);
  pActtab = acttab_alloc();
  for(i=0; i<lemp->nstate*2 && ax[i].nAction>0; i++){
    stp = ax[i].stp;
    if( ax[i].isTkn ){
      for(ap=stp->ap; ap; ap=ap->next){
        int action;
        if( ap->sp->index>=lemp->nterminal ) continue;
        action = compute_action(lemp, ap);
        if( action<0 ) continue;
        acttab_action(pActtab, ap->sp->index, action);
      }
      stp->iTknOfst = acttab_insert(pActtab);
      if( stp->iTknOfst<mnTknOfst ) mnTknOfst = stp->iTknOfst;
      if( stp->iTknOfst>mxTknOfst ) mxTknOfst = stp->iTknOfst;
    }else{
      for(ap=stp->ap; ap; ap=ap->next){
        int action;
        if( ap->sp->index<lemp->nterminal ) continue;
        if( ap->sp->index==lemp->nsymbol ) continue;
        action = compute_action(lemp, ap);
        if( action<0 ) continue;
        acttab_action(pActtab, ap->sp->index, action);
      }
      stp->iNtOfst = acttab_insert(pActtab);
      if( stp->iNtOfst<mnNtOfst ) mnNtOfst = stp->iNtOfst;
      if( stp->iNtOfst>mxNtOfst ) mxNtOfst = stp->iNtOfst;
    }
  }
  free(ax);

  /* Output the yy_action table */
  n = acttab_size(pActtab);
  fprintf(out,"#define YY_ACTTAB_COUNT (%d)\n", n); lineno++;
  fprintf(out,"static const YYACTIONTYPE yy_action[] = {\n"); lineno++;
  for(i=j=0; i<n; i++){
    int action = acttab_yyaction(pActtab, i);
    if( action<0 ) action = lemp->nstate + lemp->nrule + 2;
    if( j==0 ) fprintf(out," /* %5d */ ", i);
    fprintf(out, " %4d,", action);
    if( j==9 || i==n-1 ){
      fprintf(out, "\n"); lineno++;
      j = 0;
    }else{
      j++;
    }
  }
  fprintf(out, "};\n"); lineno++;

  /* Output the yy_lookahead table */
  fprintf(out,"static const YYCODETYPE yy_lookahead[] = {\n"); lineno++;
  for(i=j=0; i<n; i++){
    int la = acttab_yylookahead(pActtab, i);
    if( la<0 ) la = lemp->nsymbol;
    if( j==0 ) fprintf(out," /* %5d */ ", i);
    fprintf(out, " %4d,", la);
    if( j==9 || i==n-1 ){
      fprintf(out, "\n"); lineno++;
      j = 0;
    }else{
      j++;
    }
  }
  fprintf(out, "};\n"); lineno++;

  /* Output the yy_shift_ofst[] table */
  fprintf(out, "#define YY_SHIFT_USE_DFLT (%d)\n", mnTknOfst-1); lineno++;
  n = lemp->nstate;
  while( n>0 && lemp->sorted[n-1]->iTknOfst==NO_OFFSET ) n--;
  fprintf(out, "#define YY_SHIFT_COUNT (%d)\n", n-1); lineno++;
  fprintf(out, "#define YY_SHIFT_MIN   (%d)\n", mnTknOfst); lineno++;
  fprintf(out, "#define YY_SHIFT_MAX   (%d)\n", mxTknOfst); lineno++;
  fprintf(out, "static const %s yy_shift_ofst[] = {\n", 
          minimum_size_type(mnTknOfst-1, mxTknOfst)); lineno++;
  for(i=j=0; i<n; i++){
    int ofst;
    stp = lemp->sorted[i];
    ofst = stp->iTknOfst;
    if( ofst==NO_OFFSET ) ofst = mnTknOfst - 1;
    if( j==0 ) fprintf(out," /* %5d */ ", i);
    fprintf(out, " %4d,", ofst);
    if( j==9 || i==n-1 ){
      fprintf(out, "\n"); lineno++;
      j = 0;
    }else{
      j++;
    }
  }
  fprintf(out, "};\n"); lineno++;

  /* Output the yy_reduce_ofst[] table */
  fprintf(out, "#define YY_REDUCE_USE_DFLT (%d)\n", mnNtOfst-1); lineno++;
  n = lemp->nstate;
  while( n>0 && lemp->sorted[n-1]->iNtOfst==NO_OFFSET ) n--;
  fprintf(out, "#define YY_REDUCE_COUNT (%d)\n", n-1); lineno++;
  fprintf(out, "#define YY_REDUCE_MIN   (%d)\n", mnNtOfst); lineno++;
  fprintf(out, "#define YY_REDUCE_MAX   (%d)\n", mxNtOfst); lineno++;
  fprintf(out, "static const %s yy_reduce_ofst[] = {\n", 
          minimum_size_type(mnNtOfst-1, mxNtOfst)); lineno++;
  for(i=j=0; i<n; i++){
    int ofst;
    stp = lemp->sorted[i];
    ofst = stp->iNtOfst;
    if( ofst==NO_OFFSET ) ofst = mnNtOfst - 1;
    if( j==0 ) fprintf(out," /* %5d */ ", i);
    fprintf(out, " %4d,", ofst);
    if( j==9 || i==n-1 ){
      fprintf(out, "\n"); lineno++;
      j = 0;
    }else{
      j++;
    }
  }
  fprintf(out, "};\n"); lineno++;

  /* Output the default action table */
  fprintf(out, "static const YYACTIONTYPE yy_default[] = {\n"); lineno++;
  n = lemp->nstate;
  for(i=j=0; i<n; i++){
    stp = lemp->sorted[i];
    if( j==0 ) fprintf(out," /* %5d */ ", i);
    fprintf(out, " %4d,", stp->iDflt);
    if( j==9 || i==n-1 ){
      fprintf(out, "\n"); lineno++;
      j = 0;
    }else{
      j++;
    }
  }
  fprintf(out, "};\n"); lineno++;
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate the table of fallback tokens.
  */
  if( lemp->has_fallback ){
    int mx = lemp->nterminal - 1;
    while( mx>0 && lemp->symbols[mx]->fallback==0 ){ mx--; }
    for(i=0; i<=mx; i++){
      struct symbol *p = lemp->symbols[i];
      if( p->fallback==0 ){
        fprintf(out, "    0,  /* %10s => nothing */\n", p->name);
      }else{
        fprintf(out, "  %3d,  /* %10s => %s */\n", p->fallback->index,
          p->name, p->fallback->name);
      }
      lineno++;
    }
  }
  tplt_xfer(lemp->name, in, out, &lineno);

  /* Generate a table containing the symbolic name of every symbol
  */
  for(i=0; i<lemp->nsymbol; i++){
    sprintf(line,"\"%s\",",lemp->symbols[i]->name);
    fprintf(out,"  %-15s",line);
    if( (i&3)==3 ){ fprintf(out,"\n"); lineno++; }
  }
  if( (i&3)!=0 ){ fprintf(out,"\n"); lineno++; }
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate a table containing a text string that describes every
  ** rule in the rule set of the grammar.  This information is used
  ** when tracing REDUCE actions.
  */
  for(i=0, rp=lemp->rule; rp; rp=rp->next, i++){
    assert( rp->index==i );
    fprintf(out," /* %3d */ \"", i);
    writeRuleText(out, rp);
    fprintf(out,"\",\n"); lineno++;
  }
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate code which executes every time a symbol is popped from
  ** the stack while processing errors or while destroying the parser. 
  ** (In other words, generate the %destructor actions)
  */
  if( lemp->tokendest ){
    int once = 1;
    for(i=0; i<lemp->nsymbol; i++){
      struct symbol *sp = lemp->symbols[i];
      if( sp==0 || sp->type!=TERMINAL ) continue;
      if( once ){
        fprintf(out, "      /* TERMINAL Destructor */\n"); lineno++;
        once = 0;
      }
      fprintf(out,"    case %d: /* %s */\n", sp->index, sp->name); lineno++;
    }
    for(i=0; i<lemp->nsymbol && lemp->symbols[i]->type!=TERMINAL; i++);
    if( i<lemp->nsymbol ){
      emit_destructor_code(out,lemp->symbols[i],lemp,&lineno);
      fprintf(out,"      break;\n"); lineno++;
    }
  }
  if( lemp->vardest ){
    struct symbol *dflt_sp = 0;
    int once = 1;
    for(i=0; i<lemp->nsymbol; i++){
      struct symbol *sp = lemp->symbols[i];
      if( sp==0 || sp->type==TERMINAL ||
          sp->index<=0 || sp->destructor!=0 ) continue;
      if( once ){
        fprintf(out, "      /* Default NON-TERMINAL Destructor */\n"); lineno++;
        once = 0;
      }
      fprintf(out,"    case %d: /* %s */\n", sp->index, sp->name); lineno++;
      dflt_sp = sp;
    }
    if( dflt_sp!=0 ){
      emit_destructor_code(out,dflt_sp,lemp,&lineno);
    }
    fprintf(out,"      break;\n"); lineno++;
  }
  for(i=0; i<lemp->nsymbol; i++){
    struct symbol *sp = lemp->symbols[i];
    if( sp==0 || sp->type==TERMINAL || sp->destructor==0 ) continue;
    fprintf(out,"    case %d: /* %s */\n", sp->index, sp->name); lineno++;

    /* Combine duplicate destructors into a single case */
    for(j=i+1; j<lemp->nsymbol; j++){
      struct symbol *sp2 = lemp->symbols[j];
      if( sp2 && sp2->type!=TERMINAL && sp2->destructor
          && sp2->dtnum==sp->dtnum
          && strcmp(sp->destructor,sp2->destructor)==0 ){
         fprintf(out,"    case %d: /* %s */\n",
                 sp2->index, sp2->name); lineno++;
         sp2->destructor = 0;
      }
    }

    emit_destructor_code(out,lemp->symbols[i],lemp,&lineno);
    fprintf(out,"      break;\n"); lineno++;
  }
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate code which executes whenever the parser stack overflows */
  tplt_print(out,lemp,lemp->overflow,&lineno);
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate the table of rule information 
  **
  ** Note: This code depends on the fact that rules are number
  ** sequentually beginning with 0.
  */
  for(rp=lemp->rule; rp; rp=rp->next){
    fprintf(out,"  { %d, %d },\n",rp->lhs->index,rp->nrhs); lineno++;
  }
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate code which execution during each REDUCE action */
  for(rp=lemp->rule; rp; rp=rp->next){
    translate_code(lemp, rp);
  }
  /* First output rules other than the default: rule */
  for(rp=lemp->rule; rp; rp=rp->next){
    struct rule *rp2;               /* Other rules with the same action */
    if( rp->code==0 ) continue;
    if( rp->code[0]=='\n' && rp->code[1]==0 ) continue; /* Will be default: */
    fprintf(out,"      case %d: /* ", rp->index);
    writeRuleText(out, rp);
    fprintf(out, " */\n"); lineno++;
    for(rp2=rp->next; rp2; rp2=rp2->next){
      if( rp2->code==rp->code ){
        fprintf(out,"      case %d: /* ", rp2->index);
        writeRuleText(out, rp2);
        fprintf(out," */ yytestcase(yyruleno==%d);\n", rp2->index); lineno++;
        rp2->code = 0;
      }
    }
    emit_code(out,rp,lemp,&lineno);
    fprintf(out,"        break;\n"); lineno++;
    rp->code = 0;
  }
  /* Finally, output the default: rule.  We choose as the default: all
  ** empty actions. */
  fprintf(out,"      default:\n"); lineno++;
  for(rp=lemp->rule; rp; rp=rp->next){
    if( rp->code==0 ) continue;
    assert( rp->code[0]=='\n' && rp->code[1]==0 );
    fprintf(out,"      /* (%d) ", rp->index);
    writeRuleText(out, rp);
    fprintf(out, " */ yytestcase(yyruleno==%d);\n", rp->index); lineno++;
  }
  fprintf(out,"        break;\n"); lineno++;
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate code which executes if a parse fails */
  tplt_print(out,lemp,lemp->failure,&lineno);
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate code which executes when a syntax error occurs */
  tplt_print(out,lemp,lemp->error,&lineno);
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Generate code which executes when the parser accepts its input */
  tplt_print(out,lemp,lemp->accept,&lineno);
  tplt_xfer(lemp->name,in,out,&lineno);

  /* Append any addition code the user desires */
  tplt_print(out,lemp,lemp->extracode,&lineno);

  fclose(in);
  fclose(out);
  return;
}

/* Generate a header file for the parser */
void ReportHeader(struct lemon *lemp)
{
  FILE *out, *in;
  const char *prefix;
  char line[LINESIZE];
  char pattern[LINESIZE];
  int i;

  if( lemp->tokenprefix ) prefix = lemp->tokenprefix;
  else                    prefix = "";
  in = file_open(lemp,".h","rb");
  if( in ){
    for(i=1; i<lemp->nterminal && fgets(line,LINESIZE,in); i++){
      sprintf(pattern,"#define %s%-30s %2d\n",prefix,lemp->symbols[i]->name,i);
      if( strcmp(line,pattern) ) break;
    }
    fclose(in);
    if( i==lemp->nterminal ){
      /* No change in the file.  Don't rewrite it. */
      return;
    }
  }
  out = file_open(lemp,".h","wb");
  if( out ){
    for(i=1; i<lemp->nterminal; i++){
      fprintf(out,"#define %s%-30s %2d\n",prefix,lemp->symbols[i]->name,i);
    }
    fclose(out);  
  }
  return;
}
    }
}
