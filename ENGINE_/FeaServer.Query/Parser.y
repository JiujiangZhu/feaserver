%token_prefix TK
%token_type {Token}
%default_type {Token}
%extra_argument {Parse parse}
%syntax_error {
	Debug.Assert(!string.IsNullOrEmpty(TOKEN)); // The tokenizer always gives us a token
	parse.ErrorMsg("near \"%T\": syntax error", TOKEN);
}
%stack_overflow {
	parse.ErrorMsg("parser stack overflow");
}
%include {
	using FeaServer.Core;
	// An instance of this structure holds information about the LIMIT clause of a SELECT statement.
	public struct LimitValue { public Expr Limit; public Expr Offset; }
	// An instance of this structure is used to store the LIKE, GLOB, NOT LIKE, and NOT GLOB operators.
	public struct LikeOp { public Token Operator; public bool Not; }
}

// Input is a single SQL command
input ::= cmdlist.
cmdlist ::= cmdlist ecmd.
cmdlist ::= ecmd.
ecmd ::= SEMI.
ecmd ::= explain cmdx SEMI.
explain ::= . { parse.BeginParse(0); }
explain ::= EXPLAIN. { parse.BeginParse(1); }
explain ::= EXPLAIN QUERY PLAN. { parse.BeginParse(2); }
cmdx ::= cmd. { parse.FinishCoding(); }

///////////////////// Begin and end transactions. ////////////////////////////
//
cmd ::= BEGIN transtype(Y) trans_opt. { parse.BeginTransaction(Y); }
trans_opt ::= .
trans_opt ::= TRANSACTION.
trans_opt ::= TRANSACTION nm.
%type transtype {int}
transtype(A) ::= .             { A = (int)TK.DEFERRED; }
transtype(A) ::= DEFERRED(X).  { A = @X; }
transtype(A) ::= IMMEDIATE(X). { A = @X; }
transtype(A) ::= EXCLUSIVE(X). { A = @X; }
cmd ::= COMMIT trans_opt.      { parse.CommitTransaction(); }
cmd ::= END trans_opt.         { parse.CommitTransaction(); }
cmd ::= ROLLBACK trans_opt.    { parse.RollbackTransaction(); }

///////////////////// The CREATE TABLE statement ////////////////////////////
//
cmd ::= create_table create_table_args.
create_table ::= createkw temp(T) TABLE ifnotexists(E) nm(Y) dbnm(Z). { sqlite3StartTable(pParse,&Y,&Z,T,0,0,E); }
createkw(A) ::= CREATE(X). { pParse->db->lookaside.bEnabled = 0; A = X; }
%type ifnotexists {int}
ifnotexists(A) ::= .              { A = 0; }
ifnotexists(A) ::= IF NOT EXISTS. { A = 1; }
%type temp {int}
temp(A) ::= TEMP. { A = 1; }
temp(A) ::= .     { A = 0; }
create_table_args ::= LP columnlist conslist_opt(X) RP(Y). { sqlite3EndTable(pParse,&X,&Y,0); }
create_table_args ::= AS select(S). { sqlite3EndTable(pParse,0,0,S); sqlite3SelectDelete(pParse->db, S); }
columnlist ::= columnlist COMMA column.
columnlist ::= column.

// A "column" is a complete description of a single column in a CREATE TABLE statement.  This includes the column name, its
// datatype, and other keywords such as PRIMARY KEY, UNIQUE, REFERENCES, NOT NULL and so forth.
column(A) ::= columnid(X) type carglist. { A.z = X.z; A.n = (int)(pParse->sLastToken.z-X.z) + pParse->sLastToken.n; }
columnid(A) ::= nm(X). { sqlite3AddColumn(pParse,&X); A = X; }

// An IDENTIFIER can be a generic identifier, or one of several keywords.  Any non-standard keyword can also be an identifier.
%type id {Token}
id(A) ::= ID(X). { A = X; }

// The following directive causes tokens ABORT, AFTER, ASC, etc. to fallback to ID if they will not parse as their original value.
// This obviates the need for the "id" nonterminal.
%fallback ID
  ABORT
  .
%wildcard ANY.

// Define operator precedence early so that this is the first occurance of the operator tokens in the grammer.  Keeping the operators together
// causes them to be assigned integer values that are close together, which keeps parser tables smaller.
// The token values assigned to these symbols is determined by the order in which lemon first sees them.  It must be the case that ISNULL/NOTNULL,
// NE/EQ, GT/LE, and GE/LT are separated by only a single value.
%left OR.
%left AND.
%right NOT.
%left IS MATCH LIKE_KW BETWEEN IN ISNULL NOTNULL NE EQ.
%left GT LE LT GE.
%right ESCAPE.
%left BITAND BITOR LSHIFT RSHIFT.
%left PLUS MINUS.
%left STAR SLASH REM.
%left CONCAT.
%left COLLATE.
%right BITNOT.

// And "ids" is an identifer-or-string.
%type ids {Token}
ids(A) ::= ID|STRING(X). { A = X; }

// The name of a column or table can be any of the following:
%type nm {Token}
nm(A) ::= id(X).      { A = X; }
nm(A) ::= STRING(X).  { A = X; }
nm(A) ::= JOIN_KW(X). { A = X; }

// A typetoken is really one or more tokens that form a type name such as can be found after the column name in a CREATE TABLE statement.
// Multiple tokens are concatenated to form the value of the typetoken.
%type typetoken {Token}
type ::= .
type ::= typetoken(X). { sqlite3AddColumnType(pParse,&X); }
typetoken(A) ::= typename(X). { A = X; }
typetoken(A) ::= typename(X) LP signed RP(Y). { A.z = X.z; A.n = (int)(&Y.z[Y.n] - X.z); }
typetoken(A) ::= typename(X) LP signed COMMA signed RP(Y). { A.z = X.z; A.n = (int)(&Y.z[Y.n] - X.z); }
%type typename {Token}
typename(A) ::= ids(X).             {A = X;}
typename(A) ::= typename(X) ids(Y). {A.z=X.z; A.n=Y.n+(int)(Y.z-X.z);}
signed ::= plus_num.
signed ::= minus_num.

////////////////////////// The DROP TABLE /////////////////////////////////////
//
cmd ::= DROP TABLE ifexists(E) fullname(X). { sqlite3DropTable(pParse, X, 0, E); }
%type ifexists {int}
ifexists(A) ::= IF EXISTS. { A = 1; }
ifexists(A) ::= .          { A = 0; }

//////////////////////// The SELECT statement /////////////////////////////////
//
cmd ::= select(X). { SelectDest dest = {SRT_Output, 0, 0, 0, 0}; sqlite3Select(pParse, X, &dest); sqlite3SelectDelete(pParse->db, X); }
%type select {Select}
%destructor select { sqlite3SelectDelete(pParse->db, $$); }
%type oneselect {Select}
%destructor oneselect { sqlite3SelectDelete(pParse->db, $$); }
select(A) ::= oneselect(X). { A = X; }
oneselect(A) ::= SELECT selcollist(W) from(X) where_opt(Y) groupby_opt(P) having_opt(Q) orderby_opt(Z) limit_opt(L). { A = sqlite3SelectNew(pParse,W,X,Y,P,Q,Z,D,L.pLimit,L.pOffset); }
//
// selcollist is a list of expressions that are to become the return values of the SELECT statement.  The "*" in statements like
// "SELECT * FROM ..." is encoded as a special expression with an opcode of TK_ALL.
%type selcollist {ExprList}
%destructor selcollist {sqlite3ExprListDelete(pParse->db, $$);}
%type sclp {ExprList}
%destructor sclp {sqlite3ExprListDelete(pParse->db, $$);}
sclp(A) ::= selcollist(X) COMMA. { A = X; }
sclp(A) ::= .                    { A = 0; }
selcollist(A) ::= sclp(P) expr(X) as(Y). { A = sqlite3ExprListAppend(pParse, P, X.pExpr); if (Y.n > 0) sqlite3ExprListSetName(pParse, A, &Y, 1); sqlite3ExprListSetSpan(pParse,A,&X); }
selcollist(A) ::= sclp(P) STAR. { Expr p = sqlite3Expr(pParse->db, TK_ALL, 0); A = sqlite3ExprListAppend(pParse, P, p); }
selcollist(A) ::= sclp(P) nm(X) DOT STAR(Y). { Expr pRight = sqlite3PExpr(pParse, TK_ALL, 0, 0, &Y); Expr pLeft = sqlite3PExpr(pParse, TK_ID, 0, 0, &X); Expr pDot = sqlite3PExpr(pParse, TK_DOT, pLeft, pRight, 0); A = sqlite3ExprListAppend(pParse,P, pDot); }
//
// An option "AS <id>" phrase that can follow one of the expressions that define the result set, or one of the tables in the FROM clause.
%type as {Token}
as(X) ::= AS nm(Y). { X = Y; }
as(X) ::= ids(Y).   { X = Y; }
as(X) ::= .         { X.n = 0; }
//
%type seltablist {SrcList}
%destructor seltablist {sqlite3SrcListDelete(pParse->db, $$);}
%type stl_prefix {SrcList}
%destructor stl_prefix {sqlite3SrcListDelete(pParse->db, $$);}
%type from {SrcList}
%destructor from {sqlite3SrcListDelete(pParse->db, $$);}
//
// A complete FROM clause.
//
from(A) ::= . {A = sqlite3DbMallocZero(pParse->db, sizeof(A));} // sizeof(*A)
from(A) ::= FROM seltablist(X). { A = X; sqlite3SrcListShiftJoinType(A); }
// "seltablist" is a "Select Table List" - the content of the FROM clause
// in a SELECT statement.  "stl_prefix" is a prefix of this list.
stl_prefix(A) ::= seltablist(X) joinop(Y).    { A = X; if (ALWAYS(A && A->nSrc>0)) A->a[A->nSrc-1].jointype = (byte)Y; }
stl_prefix(A) ::= . { A = 0; }
seltablist(A) ::= stl_prefix(X) nm(Y) dbnm(D) as(Z) indexed_opt(I) on_opt(N) using_opt(U). { A = sqlite3SrcListAppendFromTerm(pParse,X,&Y,&D,&Z,0,N,U); sqlite3SrcListIndexedBy(pParse, A, &I); }
%type dbnm {Token}
dbnm(A) ::= .          {A.z=0; A.n=0;}
dbnm(A) ::= DOT nm(X). {A = X;}
//
%type fullname {SrcList}
%destructor fullname {sqlite3SrcListDelete(pParse->db, $$);}
fullname(A) ::= nm(X) dbnm(Y).  {A = sqlite3SrcListAppend(pParse->db,0,&X,&Y);}
//
%type joinop {int}
%type joinop2 {int}
joinop(X) ::= COMMA|JOIN.                  { X = JT_INNER; }
joinop(X) ::= JOIN_KW(A) JOIN.             { X = sqlite3JoinType(pParse,&A,0,0); }
joinop(X) ::= JOIN_KW(A) nm(B) JOIN.       { X = sqlite3JoinType(pParse,&A,&B,0); }
joinop(X) ::= JOIN_KW(A) nm(B) nm(C) JOIN. { X = sqlite3JoinType(pParse,&A,&B,&C); }
//
%type on_opt {Expr}
%destructor on_opt {sqlite3ExprDelete(pParse->db, $$);}
on_opt(N) ::= ON expr(E). { N = E.pExpr; }
on_opt(N) ::= .           { N = 0; }
//
%type indexed_opt {Token}
indexed_opt(A) ::= .                 { A.z=0; A.n=0; }
indexed_opt(A) ::= INDEXED BY nm(X). { A = X; }
indexed_opt(A) ::= NOT INDEXED.      { A.z=0; A.n=1; }
//
%type using_opt {IdList}
%destructor using_opt {sqlite3IdListDelete(pParse->db, $$);}
using_opt(U) ::= USING LP inscollist(L) RP. { U = L; }
using_opt(U) ::= .                          { U = 0; }
//
%type orderby_opt {ExprList}
%destructor orderby_opt {sqlite3ExprListDelete(pParse->db, $$);}
%type sortlist {ExprList}
%destructor sortlist {sqlite3ExprListDelete(pParse->db, $$);}
%type sortitem {Expr}
%destructor sortitem {sqlite3ExprDelete(pParse->db, $$);}
orderby_opt(A) ::= .                     { A = 0; }
orderby_opt(A) ::= ORDER BY sortlist(X). { A = X; }
sortlist(A) ::= sortlist(X) COMMA sortitem(Y) sortorder(Z). { A = sqlite3ExprListAppend(pParse,X,Y); if (A) A->a[A->nExpr-1].sortOrder = (byte)Z; }
sortlist(A) ::= sortitem(Y) sortorder(Z).                   { A = sqlite3ExprListAppend(pParse,0,Y); if (A && ALWAYS(A->a)) A->a[0].sortOrder = (byte)Z; }
sortitem(A) ::= expr(X).                                    { A = X.pExpr; }
//
%type sortorder {int}
sortorder(A) ::= ASC.  { A = SQLITE_SO_ASC; }
sortorder(A) ::= DESC. { A = SQLITE_SO_DESC; }
sortorder(A) ::= .     { A = SQLITE_SO_ASC; }
//
%type groupby_opt {ExprList}
%destructor groupby_opt {sqlite3ExprListDelete(pParse->db, $$);}
groupby_opt(A) ::= .                      { A = 0; }
groupby_opt(A) ::= GROUP BY nexprlist(X). { A = X; }
//
%type having_opt {Expr}
%destructor having_opt {sqlite3ExprDelete(pParse->db, $$);}
having_opt(A) ::= .                { A = 0; }
having_opt(A) ::= HAVING expr(X).  { A = X.pExpr; }
//
%type limit_opt {LimitVal}
limit_opt(A) ::= .                             { A.pLimit = 0; A.pOffset = 0; }
limit_opt(A) ::= LIMIT expr(X).                { A.pLimit = X.pExpr; A.pOffset = 0; }
limit_opt(A) ::= LIMIT expr(X) OFFSET expr(Y). { A.pLimit = X.pExpr; A.pOffset = Y.pExpr; }
limit_opt(A) ::= LIMIT expr(X) COMMA expr(Y).  { A.pOffset = X.pExpr; A.pLimit = Y.pExpr; }

/////////////////////////// The DELETE statement /////////////////////////////
//
cmd ::= DELETE FROM fullname(X) indexed_opt(I) where_opt(W). { sqlite3SrcListIndexedBy(pParse, X, &I); sqlite3DeleteFrom(pParse,X,W); }
%type where_opt {Expr}
%destructor where_opt {sqlite3ExprDelete(pParse->db, $$);}
where_opt(A) ::= .                    {A = 0;}
where_opt(A) ::= WHERE expr(X).       {A = X.pExpr;}

////////////////////////// The UPDATE command ////////////////////////////////
//
cmd ::= UPDATE orconf(R) fullname(X) indexed_opt(I) SET setlist(Y) where_opt(W). { sqlite3SrcListIndexedBy(pParse, X, &I); sqlite3ExprListCheckLength(pParse,Y,"set list"); sqlite3Update(pParse,X,Y,W,R); }
%type setlist {ExprList}
%destructor setlist {sqlite3ExprListDelete(pParse->db, $$);}
setlist(A) ::= setlist(Z) COMMA nm(X) EQ expr(Y). { A = sqlite3ExprListAppend(pParse, Z, Y.pExpr); sqlite3ExprListSetName(pParse, A, &X, 1); }
setlist(A) ::= nm(X) EQ expr(Y). { A = sqlite3ExprListAppend(pParse, 0, Y.pExpr); sqlite3ExprListSetName(pParse, A, &X, 1); }

////////////////////////// The INSERT command /////////////////////////////////
//
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) VALUES LP itemlist(Y) RP. {sqlite3Insert(pParse, X, Y, 0, F, R);}
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) select(S). {sqlite3Insert(pParse, X, 0, S, F, R);}
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) DEFAULT VALUES. {sqlite3Insert(pParse, X, 0, 0, F, R);}
%type insert_cmd {byte}
insert_cmd(A) ::= INSERT orconf(R). {A = R;}
insert_cmd(A) ::= REPLACE.          {A = OE_Replace;}
//
%type itemlist {ExprList}
%destructor itemlist {sqlite3ExprListDelete(pParse->db, $$);}
itemlist(A) ::= itemlist(X) COMMA expr(Y). { A = sqlite3ExprListAppend(pParse,X,Y.pExpr); }
itemlist(A) ::= expr(X). { A = sqlite3ExprListAppend(pParse,0,X.pExpr); }
//
%type inscollist_opt {IdList}
%destructor inscollist_opt {sqlite3IdListDelete(pParse->db, $$);}
%type inscollist {IdList}
%destructor inscollist {sqlite3IdListDelete(pParse->db, $$);}
inscollist_opt(A) ::= .                      { A = 0; }
inscollist_opt(A) ::= LP inscollist(X) RP.   { A = X; }
inscollist(A) ::= inscollist(X) COMMA nm(Y). { A = sqlite3IdListAppend(pParse->db,X,&Y); }
inscollist(A) ::= nm(Y).                     { A = sqlite3IdListAppend(pParse->db,0,&Y); }

/////////////////////////// Expression Processing /////////////////////////////
//
%type expr {ExprSpan}
%destructor expr { sqlite3ExprDelete(pParse->db, $$.pExpr); }
%type term {ExprSpan}
%destructor term { sqlite3ExprDelete(pParse->db, $$.pExpr); }
%include {
	// This is a utility routine used to set the ExprSpan.zStart and ExprSpan.zEnd values of pOut so that the span covers the complete range of text beginning with pStart and going to the end of pEnd.
	private static void spanSet(ExprSpan pOut, Token pStart, Token pEnd) {
		pOut->zStart = pStart->z;
		pOut->zEnd = &pEnd->z[pEnd->n];
	}
	// Construct a new Expr object from a single identifier.  Use the new Expr to populate pOut.  Set the span of pOut to be the identifier that created the expression.
	private static void spanExpr(ExprSpan pOut, Parse pParse, int op, Token pValue){
		pOut->pExpr = sqlite3PExpr(pParse, op, 0, 0, pValue);
		pOut->zStart = pValue->z;
		pOut->zEnd = &pValue->z[pValue->n];
	}
}
expr(A) ::= term(X).             { A = X; }
expr(A) ::= LP(B) expr(X) RP(E). { A.pExpr = X.pExpr; spanSet(&A,&B,&E); }
term(A) ::= NULL(X).             { spanExpr(&A, pParse, @X, &X); }
expr(A) ::= id(X).               { spanExpr(&A, pParse, TK_ID, &X); }
expr(A) ::= JOIN_KW(X).          { spanExpr(&A, pParse, TK_ID, &X); }
expr(A) ::= nm(X) DOT nm(Y). {
	Expr temp1 = sqlite3PExpr(pParse, TK_ID, 0, 0, &X);
	Expr temp2 = sqlite3PExpr(pParse, TK_ID, 0, 0, &Y);
	A.pExpr = sqlite3PExpr(pParse, TK_DOT, temp1, temp2, 0);
	spanSet(&A,&X,&Y);
}
expr(A) ::= nm(X) DOT nm(Y) DOT nm(Z). {
	Expr temp1 = sqlite3PExpr(pParse, TK_ID, 0, 0, &X);
	Expr temp2 = sqlite3PExpr(pParse, TK_ID, 0, 0, &Y);
	Expr temp3 = sqlite3PExpr(pParse, TK_ID, 0, 0, &Z);
	Expr temp4 = sqlite3PExpr(pParse, TK_DOT, temp2, temp3, 0);
	A.pExpr = sqlite3PExpr(pParse, TK_DOT, temp1, temp4, 0);
	spanSet(&A,&X,&Z);
}
term(A) ::= INTEGER|FLOAT|BLOB(X). { spanExpr(&A, pParse, @X, &X); }
term(A) ::= STRING(X).             { spanExpr(&A, pParse, @X, &X); }
expr(A) ::= REGISTER(X). {
	// When doing a nested parse, one can include terms in an expression that look like this:
	// #1 #2 ...  These terms refer to registers in the virtual machine.  #N is the N-th register.
	if (pParse->nested == 0) { sqlite3ErrorMsg(pParse, "near \"%T\": syntax error", &X); A.pExpr = 0; }
	else { A.pExpr = sqlite3PExpr(pParse, TK_REGISTER, 0, 0, &X); if (A.pExpr) sqlite3GetInt32(&X.z[1], &A.pExpr->iTable); }
	spanSet(&A, &X, &X);
}
expr(A) ::= VARIABLE(X). { spanExpr(&A, pParse, TK_VARIABLE, &X); sqlite3ExprAssignVarNumber(pParse, A.pExpr); spanSet(&A, &X, &X); }
expr(A) ::= ID(X) LP distinct(D) exprlist(Y) RP(E). {
	if (Y && Y->nExpr>pParse->db->aLimit[SQLITE_LIMIT_FUNCTION_ARG]) sqlite3ErrorMsg(pParse, "too many arguments on function %T", &X);
	A.pExpr = sqlite3ExprFunction(pParse, Y, &X);
	spanSet(&A,&X,&E);
	if (D && A.pExpr) A.pExpr->flags |= EP_Distinct;
}
expr(A) ::= ID(X) LP STAR RP(E). { A.pExpr = sqlite3ExprFunction(pParse, 0, &X); spanSet(&A,&X,&E); }
term(A) ::= CTIME_KW(OP). {
	// The CURRENT_TIME, CURRENT_DATE, and CURRENT_TIMESTAMP values are treated as functions that return constants
	A.pExpr = sqlite3ExprFunction(pParse, 0,&OP);
	if (A.pExpr) A.pExpr->op = TK_CONST_FUNC;  
	spanSet(&A, &OP, &OP);
}
%include {
	// This routine constructs a binary expression node out of two ExprSpan objects and uses the result to populate a new ExprSpan object.
	static void spanBinaryExpr(ExprSpan pOut, Parse pParse, int op, ExprSpan pLeft, ExprSpan pRight) {
		pOut->pExpr = sqlite3PExpr(pParse, op, pLeft->pExpr, pRight->pExpr, 0);
		pOut->zStart = pLeft->zStart;
		pOut->zEnd = pRight->zEnd;
	}
}
expr(A) ::= expr(X) AND(OP) expr(Y).                        { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
expr(A) ::= expr(X) OR(OP) expr(Y).                         { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
expr(A) ::= expr(X) LT|GT|GE|LE(OP) expr(Y).                { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
expr(A) ::= expr(X) EQ|NE(OP) expr(Y).                      { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
expr(A) ::= expr(X) BITAND|BITOR|LSHIFT|RSHIFT(OP) expr(Y). { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
expr(A) ::= expr(X) PLUS|MINUS(OP) expr(Y).                 { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
expr(A) ::= expr(X) STAR|SLASH|REM(OP) expr(Y).             { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
expr(A) ::= expr(X) CONCAT(OP) expr(Y).                     { spanBinaryExpr(&A,pParse,@OP,&X,&Y); }
%type likeop {LikeOp}
likeop(A) ::= LIKE_KW(X).     { A.eOperator = X; A.not = 0; }
likeop(A) ::= NOT LIKE_KW(X). { A.eOperator = X; A.not = 1; }
likeop(A) ::= MATCH(X).       { A.eOperator = X; A.not = 0; }
likeop(A) ::= NOT MATCH(X).   { A.eOperator = X; A.not = 1; }
expr(A) ::= expr(X) likeop(OP) expr(Y). [LIKE_KW] {
	ExprList pList = sqlite3ExprListAppend(pParse, 0, Y.pExpr);
	pList = sqlite3ExprListAppend(pParse, pList, X.pExpr);
	A.pExpr = sqlite3ExprFunction(pParse, pList, &OP.eOperator);
	if (OP.not) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart;
	A.zEnd = Y.zEnd;
	if (A.pExpr) A.pExpr->flags |= EP_InfixFunc;
}
expr(A) ::= expr(X) likeop(OP) expr(Y) ESCAPE expr(E). [LIKE_KW] {
	ExprList pList = sqlite3ExprListAppend(pParse, 0, Y.pExpr);
	pList = sqlite3ExprListAppend(pParse, pList, X.pExpr);
	pList = sqlite3ExprListAppend(pParse, pList, E.pExpr);
	A.pExpr = sqlite3ExprFunction(pParse, pList, &OP.eOperator);
	if (OP.not) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart;
	A.zEnd = E.zEnd;
	if (A.pExpr) A.pExpr->flags |= EP_InfixFunc;
}
//
%include {
	// Construct an expression node for a unary postfix operator
	private static void spanUnaryPostfix(ExprSpan pOut, Parse pParse, int op, ExprSpan pOperand, Token pPostOp){
		pOut->pExpr = sqlite3PExpr(pParse, op, pOperand->pExpr, 0, 0);
		pOut->zStart = pOperand->zStart;
		pOut->zEnd = &pPostOp->z[pPostOp->n];
	}
}
expr(A) ::= expr(X) ISNULL|NOTNULL(E). { spanUnaryPostfix(&A,pParse,@E,&X,&E); }
expr(A) ::= expr(X) NOT NULL(E).       { spanUnaryPostfix(&A,pParse,TK_NOTNULL,&X,&E); }
//
%include {
	// A routine to convert a binary TK_IS or TK_ISNOT expression into a unary TK_ISNULL or TK_NOTNULL expression.
	private static void binaryToUnaryIfNull(Parse pParse, Expr pY, Expr pA, int op) {
		sqlite3 db = pParse->db;
		if (db->mallocFailed == 0 && pY->op == TK_NULL) { pA->op = (byte)op; sqlite3ExprDelete(db, pA->pRight); pA->pRight = 0; }
	}
}
//    expr1 IS expr2
//    expr1 IS NOT expr2
// If expr2 is NULL then code as TK_ISNULL or TK_NOTNULL.  If expr2 is any other expression, code as TK_IS or TK_ISNOT.
expr(A) ::= expr(X) IS expr(Y).     { spanBinaryExpr(&A,pParse,TK_IS,&X,&Y); binaryToUnaryIfNull(pParse, Y.pExpr, A.pExpr, TK_ISNULL); }
expr(A) ::= expr(X) IS NOT expr(Y). { spanBinaryExpr(&A,pParse,TK_ISNOT,&X,&Y); binaryToUnaryIfNull(pParse, Y.pExpr, A.pExpr, TK_NOTNULL); }
//
%include {
	// Construct an expression node for a unary prefix operator
	private static void spanUnaryPrefix(ExprSpan Out, Parse pParse, int op, ExprSpan pOperand, Token pPreOp){
		pOut->pExpr = sqlite3PExpr(pParse, op, pOperand->pExpr, 0, 0);
		pOut->zStart = pPreOp->z;
		pOut->zEnd = pOperand->zEnd;
	}
}
expr(A) ::= NOT(B) expr(X).            { spanUnaryPrefix(&A,pParse,@B,&X,&B); }
expr(A) ::= BITNOT(B) expr(X).         { spanUnaryPrefix(&A,pParse,@B,&X,&B); }
expr(A) ::= MINUS(B) expr(X). [BITNOT] { spanUnaryPrefix(&A,pParse,TK_UMINUS,&X,&B); }
expr(A) ::= PLUS(B) expr(X). [BITNOT]  { spanUnaryPrefix(&A,pParse,TK_UPLUS,&X,&B); }
//
%type between_op {int}
between_op(A) ::= BETWEEN.     { A = 0; }
between_op(A) ::= NOT BETWEEN. { A = 1; }
expr(A) ::= expr(W) between_op(N) expr(X) AND expr(Y). [BETWEEN] {
	ExprList pList = sqlite3ExprListAppend(pParse,0, X.pExpr);
	pList = sqlite3ExprListAppend(pParse,pList, Y.pExpr);
	A.pExpr = sqlite3PExpr(pParse, TK_BETWEEN, W.pExpr, 0, 0);
	if (A.pExpr) A.pExpr->x.pList = pList;
	else sqlite3ExprListDelete(pParse->db, pList);
	if (N) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = W.zStart;
	A.zEnd = Y.zEnd;
}
// CASE expressions
expr(A) ::= CASE(C) case_operand(X) case_exprlist(Y) case_else(Z) END(E). {
	A.pExpr = sqlite3PExpr(pParse, TK_CASE, X, Z, 0);
	if (A.pExpr) { A.pExpr->x.pList = Y; sqlite3ExprSetHeight(pParse, A.pExpr); }
	else sqlite3ExprListDelete(pParse->db, Y);
	A.zStart = C.z;
	A.zEnd = &E.z[E.n];
}
%type case_exprlist {ExprList}
%destructor case_exprlist {sqlite3ExprListDelete(pParse->db, $$);}
case_exprlist(A) ::= case_exprlist(X) WHEN expr(Y) THEN expr(Z). { A = sqlite3ExprListAppend(pParse,X, Y.pExpr); A = sqlite3ExprListAppend(pParse,A, Z.pExpr); }
case_exprlist(A) ::= WHEN expr(Y) THEN expr(Z).                  { A = sqlite3ExprListAppend(pParse,0, Y.pExpr); A = sqlite3ExprListAppend(pParse,A, Z.pExpr); }
%type case_else {Expr}
%destructor case_else {sqlite3ExprDelete(pParse->db, $$);}
case_else(A) ::=  ELSE expr(X). { A = X.pExpr; }
case_else(A) ::=  .             { A = 0; }
%type case_operand {Expr}
%destructor case_operand {sqlite3ExprDelete(pParse->db, $$);}
case_operand(A) ::= expr(X).    { A = X.pExpr; } 
case_operand(A) ::= .           { A = 0; } 
//
%type exprlist {ExprList}
%destructor exprlist {sqlite3ExprListDelete(pParse->db, $$);}
%type nexprlist {ExprList}
%destructor nexprlist {sqlite3ExprListDelete(pParse->db, $$);}
exprlist(A) ::= nexprlist(X).                {A = X;}
exprlist(A) ::= .                            {A = 0;}
nexprlist(A) ::= nexprlist(X) COMMA expr(Y). {A = sqlite3ExprListAppend(pParse,X,Y.pExpr);}
nexprlist(A) ::= expr(Y).  {A = sqlite3ExprListAppend(pParse,0,Y.pExpr);}

///////////////////////////// The PRAGMA command /////////////////////////////
//
plus_num(A) ::= plus_opt number(X).   {A = X;}
minus_num(A) ::= MINUS number(X).     {A = X;}
number(A) ::= INTEGER|FLOAT(X).       {A = X;}
plus_opt ::= PLUS.
plus_opt ::= .

//////////////////////// ALTER TABLE table ... ////////////////////////////////
//
cmd ::= ALTER TABLE fullname(X) RENAME TO nm(Z). { sqlite3AlterRenameTable(pParse,X,&Z); }
cmd ::= ALTER TABLE add_column_fullname ADD kwcolumn_opt column(Y). { sqlite3AlterFinishAddColumn(pParse, &Y); }
add_column_fullname ::= fullname(X). { pParse->db->lookaside.bEnabled = 0; sqlite3AlterBeginAddColumn(pParse, X); }
kwcolumn_opt ::= .
kwcolumn_opt ::= COLUMNKW.