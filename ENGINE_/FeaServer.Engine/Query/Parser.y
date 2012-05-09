%token_prefix TK
%token_type {Token}
%default_type {Token}
%extra_argument {Parse pParse}
%syntax_error {
	Debug.Assert(minor.yy0.z.Length > 0);  /* The tokenizer always gives us a token */
	sqlite3ErrorMsg(parse, "near \"%T\": syntax error", minor.yy0);
	parse.parseError = 1;
}
%stack_overflow {
	sqlite3ErrorMsg(parse, "parser stack overflow");
	parse.parseError = 1;
}
%name Parser
%include {
	// An instance of this structure holds information about the LIMIT clause of a SELECT statement.
	public struct LimitValue { public Expr Limit; public Expr Offset; }
	// An instance of this structure is used to store the LIKE, GLOB, NOT LIKE, and NOT GLOB operators.
	public struct LikeOp { public Token Operator; public bool Not; }
	// An instance of the following structure describes the event of a TRIGGER. "a" is the event type, one of TK_UPDATE, TK_INSERT, TK_DELETE, or TK_INSTEAD.  If the event is of the form
	//    UPDATE ON (a,b,c)
	// Then the "b" IdList records the list "a,b,c".
	public struct TrigEvent { public int a; public IdList b; }
	// An instance of this structure holds the ATTACH key and the key type.
	public struct AttachKey { public int type; public Token key; }
}

// Input is a single SQL command
input ::= cmdlist.
cmdlist ::= cmdlist ecmd.
cmdlist ::= ecmd.
ecmd ::= SEMI.
ecmd ::= explain cmdx SEMI.
explain ::= .					{ sqlite3BeginParse(pParse, 0); }
explain ::= EXPLAIN.			{ sqlite3BeginParse(pParse, 1); }
explain ::= EXPLAIN QUERY PLAN. { sqlite3BeginParse(pParse, 2); }
cmdx ::= cmd.	{ sqlite3FinishCoding(pParse); }

///////////////////// Begin and end transactions. ////////////////////////////
//
cmd ::= BEGIN transtype(Y) trans_opt. { sqlite3BeginTransaction(pParse, Y); }
trans_opt ::= .
trans_opt ::= TRANSACTION.
trans_opt ::= TRANSACTION nm.
%type transtype {int}
transtype(A) ::= .             { A = TK.DEFERRED; }
transtype(A) ::= DEFERRED(X).  { A = @X; }
transtype(A) ::= IMMEDIATE(X). { A = @X; }
transtype(A) ::= EXCLUSIVE(X). { A = @X; }
cmd ::= COMMIT trans_opt.      { sqlite3CommitTransaction(pParse); }
cmd ::= END trans_opt.         { sqlite3CommitTransaction(pParse); }
cmd ::= ROLLBACK trans_opt.    { sqlite3RollbackTransaction(pParse); }
//
savepoint_opt ::= SAVEPOINT.
savepoint_opt ::= .
cmd ::= SAVEPOINT nm(X).							{ sqlite3Savepoint(pParse, SAVEPOINT_BEGIN, X); }
cmd ::= RELEASE savepoint_opt nm(X).				{ sqlite3Savepoint(pParse, SAVEPOINT_RELEASE, X); }
cmd ::= ROLLBACK trans_opt TO savepoint_opt nm(X).	{ sqlite3Savepoint(pParse, SAVEPOINT_ROLLBACK, X); }

///////////////////// The CREATE TABLE statement ////////////////////////////
//
cmd ::= create_table create_table_args.
create_table ::= createkw temp(T) TABLE ifnotexists(E) nm(Y) dbnm(Z). { sqlite3StartTable(pParse, Y, Z, T, 0, 0, E); }
createkw(A) ::= CREATE(X). { pParse.db.lookaside.bEnabled = 0; A = X; }
%type ifnotexists {int}
ifnotexists(A) ::= .				{ A = 0; }
ifnotexists(A) ::= IF NOT EXISTS.	{ A = 1; }
%type temp {int}
temp(A) ::= TEMP.	{ A = 1; }
temp(A) ::= .		{ A = 0; }
create_table_args ::= LP columnlist conslist_opt(X) RP(Y).	{ sqlite3EndTable(pParse, X, Y, 0); }
create_table_args ::= AS select(S).							{ sqlite3EndTable(pParse, 0, 0, S); sqlite3SelectDelete(pParse.db, ref S); }
columnlist ::= columnlist COMMA column.
columnlist ::= column.

// A "column" is a complete description of a single column in a CREATE TABLE statement.  This includes the column name, its
// datatype, and other keywords such as PRIMARY KEY, UNIQUE, REFERENCES, NOT NULL and so forth.
column(A) ::= columnid(X) type carglist.	{ A.n = (int)(pParse.sLastToken.z.Length - X.z.Length) + pParse.sLastToken.z.Length; A.z = X.z.Substring(0, A.n); }
columnid(A) ::= nm(X).	{ sqlite3AddColumn(pParse, X); A = X; }

// An IDENTIFIER can be a generic identifier, or one of several keywords.  Any non-standard keyword can also be an identifier.
%type id {Token}
id(A) ::= ID(X).		{ A = X; }
id(A) ::= INDEXED(X).	{ A = X; }

// The following directive causes tokens ABORT, AFTER, ASC, etc. to fallback to ID if they will not parse as their original value.
// This obviates the need for the "id" nonterminal.
%fallback ID
  ABORT ACTION AFTER ANALYZE ASC ATTACH BEFORE BEGIN BY CASCADE CAST COLUMNKW
  CONFLICT DATABASE DEFERRED DESC DETACH EACH END EXCLUSIVE EXPLAIN FAIL FOR
  IGNORE IMMEDIATE INITIALLY INSTEAD LIKE_KW MATCH NO PLAN
  QUERY KEY OF OFFSET PRAGMA RAISE RELEASE REPLACE RESTRICT ROW ROLLBACK
  SAVEPOINT TEMP TRIGGER VACUUM VIEW VIRTUAL
  EXCEPT INTERSECT UNION
  REINDEX RENAME CTIME_KW IF
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
type ::= typetoken(X).	{ sqlite3AddColumnType(pParse, X); }
typetoken(A) ::= typename(X).								{ A = X; }
typetoken(A) ::= typename(X) LP signed RP(Y).				{ A.n = Y.z.Length - X.z.Length + X.n; A.z = X.z.Substring(0, A.n); }
typetoken(A) ::= typename(X) LP signed COMMA signed RP(Y).	{ A.n = Y.z.Length - X.z.Length + X.n; A.z = X.z.Substring(0, A.n); }
%type typename {Token}
typename(A) ::= ids(X).				{ A = X; }
typename(A) ::= typename(X) ids(Y). { A.z = X.z; A.n = Y.n + Y.z.Length - X.z.Length;  }
signed ::= plus_num.
signed ::= minus_num.

// "carglist" is a list of additional constraints that come after the column name and column type in a CREATE TABLE statement.
carglist ::= carglist carg.
carglist ::= .
carg ::= CONSTRAINT nm ccons.
carg ::= ccons.
ccons ::= DEFAULT term(X).          { sqlite3AddDefaultValue(pParse, X); }
ccons ::= DEFAULT LP expr(X) RP.    { sqlite3AddDefaultValue(pParse, X); }
ccons ::= DEFAULT PLUS term(X).		{ sqlite3AddDefaultValue(pParse, X); }
ccons ::= DEFAULT MINUS(A) term(X). { var v = new ExprSpan { pExpr = sqlite3PExpr(pParse, TK.UMINUS, X.pExpr, 0, 0), zStart = A.z, zEnd = X.zEnd }; sqlite3AddDefaultValue(pParse, v); }
ccons ::= DEFAULT id(X).			{ var v = new ExprSpan(); spanExpr(v, pParse, TK.STRING, X); sqlite3AddDefaultValue(pParse, v); }
//
// In addition to the type name, we also care about the primary key and UNIQUE constraints.
ccons ::= NULL onconf.
ccons ::= NOT NULL onconf(R).		{ sqlite3AddNotNull(pParse, R); }
ccons ::= PRIMARY KEY sortorder(Z) onconf(R) autoinc(I). { sqlite3AddPrimaryKey(pParse, 0, R, I, Z); }
ccons ::= UNIQUE onconf(R).			{ sqlite3CreateIndex(pParse, 0, 0, 0, 0, R, 0, 0, 0, 0);}
ccons ::= CHECK LP expr(X) RP.		{ sqlite3AddCheckConstraint(pParse, X.pExpr);}
ccons ::= REFERENCES nm(T) idxlist_opt(TA) refargs(R). { sqlite3CreateForeignKey(pParse, 0, T, TA, R); }
ccons ::= defer_subclause(D).		{ sqlite3DeferForeignKey(pParse, D); }
ccons ::= COLLATE ids(C).			{ sqlite3AddCollateType(pParse, C); }
//
// The optional AUTOINCREMENT keyword
%type autoinc {int}
autoinc(X) ::= .			{ X = 0; }
autoinc(X) ::= AUTOINCR.	{ X = 1; }
//
// The next group of rules parses the arguments to a REFERENCES clause that determine if the referential integrity checking is deferred or
// or immediate and which determine what action to take if a ref-integ check fails.
%type refargs {int}
refargs(A) ::= .					{ A = OE_None * 0x0101; }
refargs(A) ::= refargs(X) refarg(Y).{ A = (X & ~Y.mask) | Y.value; }
%type refarg {public struct { public int value; public int mask; }}
refarg(A) ::= MATCH nm.             { A.value = 0;    A.mask = 0x000000; }
refarg(A) ::= ON INSERT refact.     { A.value = 0;    A.mask = 0x000000; }
refarg(A) ::= ON DELETE refact(X).  { A.value = X;    A.mask = 0x0000ff; }
refarg(A) ::= ON UPDATE refact(X).  { A.value = X<<8; A.mask = 0x00ff00; }
%type refact {int}
refact(A) ::= SET NULL.             { A = OE_SetNull; }
refact(A) ::= SET DEFAULT.          { A = OE_SetDflt; }
refact(A) ::= CASCADE.              { A = OE_Cascade; }
refact(A) ::= RESTRICT.             { A = OE_Restrict; }
refact(A) ::= NO ACTION.            { A = OE_None; }
%type defer_subclause {int}
defer_subclause(A) ::= NOT DEFERRABLE init_deferred_pred_opt.	{ A = 0; }
defer_subclause(A) ::= DEFERRABLE init_deferred_pred_opt(X).	{ A = X; }
%type init_deferred_pred_opt {int}
init_deferred_pred_opt(A) ::= .                     { A = 0; }
init_deferred_pred_opt(A) ::= INITIALLY DEFERRED.   { A = 1; }
init_deferred_pred_opt(A) ::= INITIALLY IMMEDIATE.  { A = 0; }
//
// For the time being, the only constraint we care about is the primary key and UNIQUE.  Both create indices.
conslist_opt(A) ::= .                   { A.n = 0; A.z = null; }
conslist_opt(A) ::= COMMA(X) conslist.	{ A = X; }
conslist ::= conslist COMMA tcons.
conslist ::= conslist tcons.
conslist ::= tcons.
tcons ::= CONSTRAINT nm.
tcons ::= PRIMARY KEY LP idxlist(X) autoinc(I) RP onconf(R).	{ sqlite3AddPrimaryKey(pParse, X, R, I, 0); }
tcons ::= UNIQUE LP idxlist(X) RP onconf(R).					{ sqlite3CreateIndex(pParse, 0, 0, 0, X, R, 0, 0, 0, 0); }
tcons ::= CHECK LP expr(E) RP onconf.							{ sqlite3AddCheckConstraint(pParse, E.pExpr); }
tcons ::= FOREIGN KEY LP idxlist(FA) RP REFERENCES nm(T) idxlist_opt(TA) refargs(R) defer_subclause_opt(D). { sqlite3CreateForeignKey(pParse, FA, T, TA, R); sqlite3DeferForeignKey(pParse, D); }
%type defer_subclause_opt {int}
defer_subclause_opt(A) ::= .                    { A = 0; }
defer_subclause_opt(A) ::= defer_subclause(X).	{ A = X; }
//
// The following is a non-standard extension that allows us to declare the default behavior when there is a constraint conflict.
%type onconf {int}
%type orconf {u8}
%type resolvetype {int}
onconf(A) ::= .                             { A = OE_Default; }
onconf(A) ::= ON CONFLICT resolvetype(X).   { A = X; }
orconf(A) ::= .								{ A = OE_Default; }
orconf(A) ::= OR resolvetype(X).			{ A = (byte)X;}
resolvetype(A) ::= raisetype(X).			{ A = X; }
resolvetype(A) ::= IGNORE.					{ A = OE_Ignore; }
resolvetype(A) ::= REPLACE.					{ A = OE_Replace; }


////////////////////////// The DROP TABLE /////////////////////////////////////
//
cmd ::= DROP TABLE ifexists(E) fullname(X). { sqlite3DropTable(pParse, X, 0, E); }
%type ifexists {int}
ifexists(A) ::= IF EXISTS. { A = 1; }
ifexists(A) ::= .          { A = 0; }


//////////////////////// The SELECT statement /////////////////////////////////
//
cmd ::= select(X). { var dest = new SelectDest(SRT_Output, '\0', 0, 0, 0}; sqlite3Select(pParse, X, ref dest); sqlite3SelectDelete(pParse.db, ref X); }
%type select {Select}
%destructor select { sqlite3SelectDelete(pParse.db, $$); }
%type oneselect {Select}
%destructor oneselect { sqlite3SelectDelete(pParse.db, $$); }
select(A) ::= oneselect(X). { A = X; }
select(A) ::= select(X) multiselect_op(Y) oneselect(Z). {
	if (Z != null) { Z->op = (TK)Y; Z->pPrior = X; }
	else { sqlite3SelectDelete(pParse.db, ref X); }
	A = Z; }
%type multiselect_op {int}
multiselect_op(A) ::= UNION(OP).            { A = @OP; }
multiselect_op(A) ::= UNION ALL.            { A = TK.ALL; }
multiselect_op(A) ::= EXCEPT|INTERSECT(OP). { A = @OP; }
oneselect(A) ::= SELECT selcollist(W) from(X) where_opt(Y) groupby_opt(P) having_opt(Q) orderby_opt(Z) limit_opt(L). { A = sqlite3SelectNew(pParse, W, X, Y, P, Q, Z, D, L.pLimit, L.pOffset); }
//
// The "distinct" nonterminal is true (1) if the DISTINCT keyword is present and false (0) if it is not.
%type distinct {int}
distinct(A) ::= DISTINCT.	{ A = 1; }
distinct(A) ::= ALL.        { A = 0; }
distinct(A) ::= .           { A = 0; }
//
// selcollist is a list of expressions that are to become the return values of the SELECT statement.  The "*" in statements like
// "SELECT * FROM ..." is encoded as a special expression with an opcode of TK_ALL.
%type selcollist {ExprList}
%destructor selcollist { sqlite3ExprListDelete(pParse.db, $$); }
%type sclp {ExprList}
%destructor sclp { sqlite3ExprListDelete(pParse.db, $$); }
sclp(A) ::= selcollist(X) COMMA.	{ A = X; }
sclp(A) ::= .						{ A = null; }
selcollist(A) ::= sclp(P) expr(X) as(Y).	{ A = sqlite3ExprListAppend(pParse, P, X.pExpr); if (Y.n > 0) sqlite3ExprListSetName(pParse, A, Y, 1); sqlite3ExprListSetSpan(pParse, A, X); }
selcollist(A) ::= sclp(P) STAR.				{ var p = sqlite3Expr(pParse.db, TK.ALL, null); A = sqlite3ExprListAppend(pParse, P, p); }
selcollist(A) ::= sclp(P) nm(X) DOT STAR(Y).{ var pRight = sqlite3PExpr(pParse, TK.ALL, 0, 0, Y); var pLeft = sqlite3PExpr(pParse, TK.ID, 0, 0, X); var pDot = sqlite3PExpr(pParse, TK.DOT, pLeft, pRight, 0); A = sqlite3ExprListAppend(pParse, P, pDot); }
//
// An option "AS <id>" phrase that can follow one of the expressions that define the result set, or one of the tables in the FROM clause.
%type as {Token}
as(X) ::= AS nm(Y).	{ X = Y; }
as(X) ::= ids(Y).   { X = Y; }
as(X) ::= .         { X.n = 0; }
//
%type seltablist {SrcList}
%destructor seltablist { sqlite3SrcListDelete(pParse.db, $$); }
%type stl_prefix {SrcList}
%destructor stl_prefix { sqlite3SrcListDelete(pParse.db, $$); }
%type from {SrcList}
%destructor from { sqlite3SrcListDelete(pParse.db, $$); }
//
// A complete FROM clause.
from(A) ::= .					{ A = new SrcList(); }
from(A) ::= FROM seltablist(X). { A = X; sqlite3SrcListShiftJoinType(A); }
//
// "seltablist" is a "Select Table List" - the content of the FROM clause in a SELECT statement.  "stl_prefix" is a prefix of this list.
stl_prefix(A) ::= seltablist(X) joinop(Y).  { A = X; if (Sanity.ALWAYS(A != null && A.nSrc > 0)) A.a[A.nSrc - 1].jointype = (byte)Y; }
stl_prefix(A) ::= .							{ A = null; }
seltablist(A) ::= stl_prefix(X) nm(Y) dbnm(D) as(Z) indexed_opt(I) on_opt(N) using_opt(U).	{ A = sqlite3SrcListAppendFromTerm(pParse, X, Y, D, Z, 0, N, U); sqlite3SrcListIndexedBy(pParse, A, I); }
seltablist(A) ::= stl_prefix(X) LP select(S) RP as(Z) on_opt(N) using_opt(U).				{ A = sqlite3SrcListAppendFromTerm(pParse, X, 0, 0, Z, S, N, U); }
seltablist(A) ::= stl_prefix(X) LP seltablist(F) RP as(Z) on_opt(N) using_opt(U). {
	if (X == null && Z.n == 0 && N == null && U == null) { A = F; }
	else { sqlite3SrcListShiftJoinType(F); var pSubquery = sqlite3SelectNew(pParse, 0, F, 0, 0, 0, 0, 0, 0, 0); A = sqlite3SrcListAppendFromTerm(pParse, X, 0, 0, Z, pSubquery, N, U); } }
//
%type dbnm {Token}
dbnm(A) ::= .			{ A.z = null; A.n = 0; }
dbnm(A) ::= DOT nm(X).	{ A = X; }
//
%type fullname {SrcList}
%destructor fullname { sqlite3SrcListDelete(pParse.db, $$); }
fullname(A) ::= nm(X) dbnm(Y).  { A = sqlite3SrcListAppend(pParse.db, 0, X, Y); }
//
%type joinop {int}
%type joinop2 {int}
joinop(X) ::= COMMA|JOIN.					{ X = JT_INNER; }
joinop(X) ::= JOIN_KW(A) JOIN.				{ X = sqlite3JoinType(pParse, A, 0, 0); }
joinop(X) ::= JOIN_KW(A) nm(B) JOIN.		{ X = sqlite3JoinType(pParse, A, B, 0); }
joinop(X) ::= JOIN_KW(A) nm(B) nm(C) JOIN.	{ X = sqlite3JoinType(pParse, A, B, C); }
//
%type on_opt {Expr}
%destructor on_opt { sqlite3ExprDelete(pParse.db, $$); }
on_opt(N) ::= ON expr(E).	{ N = E.pExpr; }
on_opt(N) ::= .				{ N = null; }
//
%type indexed_opt {Token}
indexed_opt(A) ::= .					{ A.z = null; A.n = 0; }
indexed_opt(A) ::= INDEXED BY nm(X).	{ A = X; }
indexed_opt(A) ::= NOT INDEXED.			{ A.z = null; A.n = 1; }
//
%type using_opt {IdList}
%destructor using_opt { sqlite3IdListDelete(pParse.db, $$); }
using_opt(U) ::= USING LP inscollist(L) RP.	{ U = L; }
using_opt(U) ::= .                          { U = null; }
//
%type orderby_opt {ExprList}
%destructor orderby_opt { sqlite3ExprListDelete(pParse.db, $$); }
%type sortlist {ExprList}
%destructor sortlist { sqlite3ExprListDelete(pParse.db, $$); }
%type sortitem {Expr}
%destructor sortitem { sqlite3ExprDelete(pParse.db, $$); }
orderby_opt(A) ::= .						{ A = null; }
orderby_opt(A) ::= ORDER BY sortlist(X).	{ A = X; }
sortlist(A) ::= sortlist(X) COMMA sortitem(Y) sortorder(Z).	{ A = sqlite3ExprListAppend(pParse, X, Y); if (A != null) A->a[A->nExpr-1].sortOrder = (u8)Z; }
sortlist(A) ::= sortitem(Y) sortorder(Z).                   { A = sqlite3ExprListAppend(pParse, 0, Y); if (A != null && Sanity.ALWAYS(A.a != null)) A.a[0].sortOrder = (byte)Z; }
sortitem(A) ::= expr(X).                                    { A = X.pExpr; }
//
%type sortorder {int}
sortorder(A) ::= ASC.	{ A = SQLITE_SO_ASC; }
sortorder(A) ::= DESC.	{ A = SQLITE_SO_DESC; }
sortorder(A) ::= .		{ A = SQLITE_SO_ASC; }
//
%type groupby_opt {ExprList}
%destructor groupby_opt { sqlite3ExprListDelete(pParse.db, $$); }
groupby_opt(A) ::= .						{ A = 0; }
groupby_opt(A) ::= GROUP BY nexprlist(X).	{ A = X; }
//
%type having_opt {Expr}
%destructor having_opt { sqlite3ExprDelete(pParse.db, $$); }
having_opt(A) ::= .					{ A = 0; }
having_opt(A) ::= HAVING expr(X).	{ A = X.pExpr; }
//
%type limit_opt {LimitVal}
limit_opt(A) ::= .								{ A.pLimit = null; A.pOffset = null; }
limit_opt(A) ::= LIMIT expr(X).					{ A.pLimit = X.pExpr; A.pOffset = null; }
limit_opt(A) ::= LIMIT expr(X) OFFSET expr(Y).	{ A.pLimit = X.pExpr; A.pOffset = Y.pExpr; }
limit_opt(A) ::= LIMIT expr(X) COMMA expr(Y).	{ A.pOffset = X.pExpr; A.pLimit = Y.pExpr; }


/////////////////////////// The DELETE statement /////////////////////////////
//
cmd ::= DELETE FROM fullname(X) indexed_opt(I) where_opt(W). { sqlite3SrcListIndexedBy(pParse, X, I); sqlite3DeleteFrom(pParse, X, W); }
%type where_opt {Expr}
%destructor where_opt { sqlite3ExprDelete(pParse.db, $$); }
where_opt(A) ::= .              { A = null; }
where_opt(A) ::= WHERE expr(X).	{ A = X.pExpr; }


////////////////////////// The UPDATE command ////////////////////////////////
//
cmd ::= UPDATE orconf(R) fullname(X) indexed_opt(I) SET setlist(Y) where_opt(W). { 
	sqlite3SrcListIndexedBy(pParse, X, I); 
	sqlite3ExprListCheckLength(pParse, Y, "set list"); 
	sqlite3Update(pParse, X, Y, W, R); }
%type setlist {ExprList}
%destructor setlist { sqlite3ExprListDelete(pParse.db, $$); }
setlist(A) ::= setlist(Z) COMMA nm(X) EQ expr(Y).	{ A = sqlite3ExprListAppend(pParse, Z, Y.pExpr); sqlite3ExprListSetName(pParse, A, X, 1); }
setlist(A) ::= nm(X) EQ expr(Y).					{ A = sqlite3ExprListAppend(pParse, 0, Y.pExpr); sqlite3ExprListSetName(pParse, A, X, 1); }


////////////////////////// The INSERT command /////////////////////////////////
//
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) VALUES LP itemlist(Y) RP.	{ sqlite3Insert(pParse, X, Y, 0, F, R); }
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) select(S).					{ sqlite3Insert(pParse, X, 0, S, F, R); }
cmd ::= insert_cmd(R) INTO fullname(X) inscollist_opt(F) DEFAULT VALUES.			{ sqlite3Insert(pParse, X, 0, 0, F, R); }
%type insert_cmd {byte}
insert_cmd(A) ::= INSERT orconf(R).	{ A = R; }
insert_cmd(A) ::= REPLACE.          { A = OE_Replace; }
//
%type itemlist {ExprList}
%destructor itemlist { sqlite3ExprListDelete(pParse.db, $$); }
itemlist(A) ::= itemlist(X) COMMA expr(Y).	{ A = sqlite3ExprListAppend(pParse, X, Y.pExpr); }
itemlist(A) ::= expr(X).					{ A = sqlite3ExprListAppend(pParse, 0, X.pExpr); }
//
%type inscollist_opt {IdList}
%destructor inscollist_opt { sqlite3IdListDelete(pParse.db, $$); }
%type inscollist {IdList}
%destructor inscollist { sqlite3IdListDelete(pParse.db, $$); }
inscollist_opt(A) ::= .							{ A = null; }
inscollist_opt(A) ::= LP inscollist(X) RP.		{ A = X; }
inscollist(A) ::= inscollist(X) COMMA nm(Y).	{ A = sqlite3IdListAppend(pParse.db, X, Y); }
inscollist(A) ::= nm(Y).						{ A = sqlite3IdListAppend(pParse.db, 0, Y); }


/////////////////////////// Expression Processing /////////////////////////////
//
%type expr {ExprSpan}
%destructor expr { sqlite3ExprDelete(pParse.db, $$.pExpr); }
%type term {ExprSpan}
%destructor term { sqlite3ExprDelete(pParse.db, $$.pExpr); }
%include {
	// This is a utility routine used to set the ExprSpan.zStart and ExprSpan.zEnd values of pOut so that the span covers the complete range of text beginning with pStart and going to the end of pEnd.
	private static void spanSet(ExprSpan pOut, Token pStart, Token pEnd) { pOut.zStart = pStart.z; pOut.zEnd = pEnd.z.Substring(pEnd.n); }
	// Construct a new Expr object from a single identifier.  Use the new Expr to populate pOut.  Set the span of pOut to be the identifier that created the expression.
	private static void spanExpr(ExprSpan pOut, Parse pParse, int op, Token pValue) { pOut->pExpr = sqlite3PExpr(pParse, op, 0, 0, pValue); pOut->zStart = pValue.z; pOut->zEnd = pValue.z.Substring(pValue.n); }
}
//
expr(A) ::= term(X).				{ A = X; }
expr(A) ::= LP(B) expr(X) RP(E).	{ A.pExpr = X.pExpr; spanSet(A, B, E); }
term(A) ::= NULL(X).				{ spanExpr(A, pParse, @X, X); }
expr(A) ::= id(X).					{ spanExpr(A, pParse, TK.ID, X); }
expr(A) ::= JOIN_KW(X).				{ spanExpr(A, pParse, TK.ID, X); }
expr(A) ::= nm(X) DOT nm(Y). {
	var temp1 = sqlite3PExpr(pParse, TK.ID, 0, 0, X);
	var temp2 = sqlite3PExpr(pParse, TK.ID, 0, 0, Y);
	A.pExpr = sqlite3PExpr(pParse, TK.DOT, temp1, temp2, 0);
	spanSet(A, X, Y); }
expr(A) ::= nm(X) DOT nm(Y) DOT nm(Z). {
	var temp1 = sqlite3PExpr(pParse, TK.ID, 0, 0, X);
	var temp2 = sqlite3PExpr(pParse, TK.ID, 0, 0, Y);
	var temp3 = sqlite3PExpr(pParse, TK.ID, 0, 0, Z);
	var temp4 = sqlite3PExpr(pParse, TK.DOT, temp2, temp3, 0);
	A.pExpr = sqlite3PExpr(pParse, TK.DOT, temp1, temp4, 0);
	spanSet(A, X, Z); }
term(A) ::= INTEGER|FLOAT|BLOB(X).	{ spanExpr(A, pParse, @X, X); }
term(A) ::= STRING(X).				{ spanExpr(A, pParse, @X, X); }
expr(A) ::= REGISTER(X). {
	// When doing a nested parse, one can include terms in an expression that look like this:
	// #1 #2 ...  These terms refer to registers in the virtual machine.  #N is the N-th register. */
	if (pParse.nested == 0) { sqlite3ErrorMsg(pParse, "near \"%T\": syntax error", X); A.pExpr = null; }
	else { A.pExpr = sqlite3PExpr(pParse, TK.REGISTER, 0, 0, X); if (A.pExpr != null) sqlite3GetInt32(X.z, 1, ref A.pExpr.iTable); }
	spanSet(A, X, X); }
expr(A) ::= VARIABLE(X). { spanExpr(A, pParse, TK.VARIABLE, X); sqlite3ExprAssignVarNumber(pParse, A.pExpr); spanSet(A, X, X); }
expr(A) ::= expr(E) COLLATE ids(C). { A.pExpr = sqlite3ExprSetCollByToken(pParse, E.pExpr, C); A.zStart = E.zStart; A.zEnd = C.z.Substring(C.n); }
expr(A) ::= CAST(X) LP expr(E) AS typetoken(T) RP(Y). { A.pExpr = sqlite3PExpr(pParse, TK.CAST, E.pExpr, 0, T); spanSet(A, X, Y); }
expr(A) ::= ID(X) LP distinct(D) exprlist(Y) RP(E). {
	if (Y != null && Y.nExpr > pParse.db.aLimit[SQLITE_LIMIT_FUNCTION_ARG]) sqlite3ErrorMsg(pParse, "too many arguments on function %T", X);
	A.pExpr = sqlite3ExprFunction(pParse, Y, X);
	spanSet(A, X, E);
	if (D != 0 && A.pExpr != null) A.pExpr->flags |= EP_Distinct; }
expr(A) ::= ID(X) LP STAR RP(E).	{ A.pExpr = sqlite3ExprFunction(pParse, 0, X); spanSet(A, X, E); }
term(A) ::= CTIME_KW(OP). {
	// The CURRENT_TIME, CURRENT_DATE, and CURRENT_TIMESTAMP values are treated as functions that return constants
	A.pExpr != null) A.pExpr.op = TK.CONST_FUNC;  
	spanSet(A, OP, OP); }
//
%include {
	// This routine constructs a binary expression node out of two ExprSpan objects and uses the result to populate a new ExprSpan object.
	private static void spanBinaryExpr(ExprSpan pOut, Parse pParse, int op, ExprSpan pLeft, ExprSpan pRight) { pOut.pExpr = sqlite3PExpr(pParse, op, pLeft.pExpr, pRight.pExpr, 0); pOut.zStart = pLeft.zStart; pOut.zEnd = pRight.zEnd; }
}
expr(A) ::= expr(X) AND(OP) expr(Y).                        { spanBinaryExpr(A, pParse, @OP, X, Y); }
expr(A) ::= expr(X) OR(OP) expr(Y).                         { spanBinaryExpr(A, pParse, @OP, X, Y); }
expr(A) ::= expr(X) LT|GT|GE|LE(OP) expr(Y).                { spanBinaryExpr(A, pParse, @OP, X, Y); }
expr(A) ::= expr(X) EQ|NE(OP) expr(Y).                      { spanBinaryExpr(A, pParse, @OP, X, Y); }
expr(A) ::= expr(X) BITAND|BITOR|LSHIFT|RSHIFT(OP) expr(Y). { spanBinaryExpr(A, pParse, @OP, X, Y); }
expr(A) ::= expr(X) PLUS|MINUS(OP) expr(Y).                 { spanBinaryExpr(A, pParse, @OP, X, Y); }
expr(A) ::= expr(X) STAR|SLASH|REM(OP) expr(Y).             { spanBinaryExpr(A, pParse, @OP, X, Y); }
expr(A) ::= expr(X) CONCAT(OP) expr(Y).                     { spanBinaryExpr(A, pParse, @OP, X, Y); }
%type likeop {LikeOp}
likeop(A) ::= LIKE_KW(X).     { A.eOperator = X; A.not = false; }
likeop(A) ::= NOT LIKE_KW(X). { A.eOperator = X; A.not = true; }
likeop(A) ::= MATCH(X).       { A.eOperator = X; A.not = false; }
likeop(A) ::= NOT MATCH(X).   { A.eOperator = X; A.not = true; }
expr(A) ::= expr(X) likeop(OP) expr(Y). [LIKE_KW] {
	var pList = sqlite3ExprListAppend(pParse, 0, Y.pExpr);
	pList = sqlite3ExprListAppend(pParse, pList, X.pExpr);
	A.pExpr = sqlite3ExprFunction(pParse, pList, OP.eOperator);
	if (OP.not) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart; A.zEnd = Y.zEnd;
	if (A.pExpr != null) A.pExpr.flags |= EP_InfixFunc; }
expr(A) ::= expr(X) likeop(OP) expr(Y) ESCAPE expr(E). [LIKE_KW] {
	var pList = sqlite3ExprListAppend(pParse, 0, Y.pExpr);
	pList = sqlite3ExprListAppend(pParse, pList, X.pExpr);
	pList = sqlite3ExprListAppend(pParse, pList, E.pExpr);
	A.pExpr = sqlite3ExprFunction(pParse, pList, OP.eOperator);
	if (OP.not) A.pExpr = sqlite3PExpr(pParse, TK_NOT, A.pExpr, 0, 0);
	A.zStart = X.zStart; A.zEnd = E.zEnd;
	if (A.pExpr != null) A.pExpr.flags |= EP_InfixFunc; }
//
%include {
	// Construct an expression node for a unary postfix operator
	private static void spanUnaryPostfix(ExprSpan pOut, Parse pParse, int op, ExprSpan pOperand, Token pPostOp) { pOut.pExpr = sqlite3PExpr(pParse, op, pOperand.pExpr, 0, 0); pOut.zStart = pOperand.zStart; pOut.zEnd = pPostOp.z.Substring(pPostOp.n); }
}
expr(A) ::= expr(X) ISNULL|NOTNULL(E).	{ spanUnaryPostfix(A, pParse, @E, X, E); }
expr(A) ::= expr(X) NOT NULL(E).		{ spanUnaryPostfix(A, pParse, TK.NOTNULL, X, E); }
//
%include {
	// A routine to convert a binary TK_IS or TK_ISNOT expression into a unary TK_ISNULL or TK_NOTNULL expression.
	private static void binaryToUnaryIfNull(Parse pParse, Expr pY, Expr pA, int op) { var db = pParse.db; if (db.mallocFailed == 0 && pY.op == TK.NULL) { pA.op = (TK)op; sqlite3ExprDelete(db, pA.pRight); pA.pRight = null; } }
}
//    expr1 IS expr2
//    expr1 IS NOT expr2
// If expr2 is NULL then code as TK_ISNULL or TK_NOTNULL.  If expr2 is any other expression, code as TK_IS or TK_ISNOT.
expr(A) ::= expr(X) IS expr(Y).			{ spanBinaryExpr(A, pParse, TK.IS, X, Y); binaryToUnaryIfNull(pParse, Y.pExpr, A.pExpr, TK.ISNULL); }
expr(A) ::= expr(X) IS NOT expr(Y).		{ spanBinaryExpr(A, pParse, TK.ISNOT, X, Y); binaryToUnaryIfNull(pParse, Y.pExpr, A.pExpr, TK.NOTNULL); }
//
%include {
	// Construct an expression node for a unary prefix operator
	private static void spanUnaryPrefix(ExprSpan pOut, Parse pParse, int op, ExprSpan pOperand, Token pPreOp) { pOut.pExpr = sqlite3PExpr(pParse, op, pOperand.pExpr, 0, 0); pOut.zStart = pPreOp.z; pOut.zEnd = pOperand.zEnd; }
}
expr(A) ::= NOT(B) expr(X).				{ spanUnaryPrefix(A, pParse, @B, X, B); }
expr(A) ::= BITNOT(B) expr(X).			{ spanUnaryPrefix(A, pParse, @B, X, B); }
expr(A) ::= MINUS(B) expr(X). [BITNOT]	{ spanUnaryPrefix(A, pParse, TK.UMINUS, X, B); }
expr(A) ::= PLUS(B) expr(X). [BITNOT]	{ spanUnaryPrefix(A, pParse, TK.UPLUS, X, B); }
//
%type between_op {int}
between_op(A) ::= BETWEEN.		{ A = 0; }
between_op(A) ::= NOT BETWEEN.	{ A = 1; }
expr(A) ::= expr(W) between_op(N) expr(X) AND expr(Y). [BETWEEN] {
	var pList = sqlite3ExprListAppend(pParse, 0, X.pExpr);
	pList = sqlite3ExprListAppend(pParse, pList, Y.pExpr);
	A.pExpr = sqlite3PExpr(pParse, TK.BETWEEN, W.pExpr, 0, 0);
	if (A.pExpr != null) A.pExpr.x.pList = pList;
	else sqlite3ExprListDelete(pParse.db, pList);
	if (N != 0) A.pExpr = sqlite3PExpr(pParse, TK.NOT, A.pExpr, 0, 0);
	A.zStart = W.zStart; A.zEnd = Y.zEnd;
}
//
%type in_op {int}
in_op(A) ::= IN.		{ A = 0; }
in_op(A) ::= NOT IN.	{ A = 1; }
expr(A) ::= expr(X) in_op(N) LP exprlist(Y) RP(E). [IN] {
	if (Y == null) {
		// Expressions of the form
		//      expr1 IN ()
		//      expr1 NOT IN ()
		// simplify to constants 0 (false) and 1 (true), respectively,  regardless of the value of expr1.
		A.pExpr = sqlite3PExpr(pParse, TK.INTEGER, 0, 0, sqlite3IntTokens[N]);
		sqlite3ExprDelete(pParse.db, X.pExpr);
	} else {
		A.pExpr = sqlite3PExpr(pParse, TK.IN, X.pExpr, 0, 0);
		if (A.pExpr != null) { A.pExpr.x.pList = Y; sqlite3ExprSetHeight(pParse, A.pExpr); }
		else sqlite3ExprListDelete(pParse.db, ref Y);
		if (N != 0) A.pExpr = sqlite3PExpr(pParse, TK.NOT, A.pExpr, 0, 0);
    }
    A.zStart = X.zStart; A.zEnd = E.z.Substring(E.n); }
expr(A) ::= LP(B) select(X) RP(E). {
	A.pExpr = sqlite3PExpr(pParse, TK.SELECT, 0, 0, 0);
    if (A.pExpr != null) { A.pExpr.x.pSelect = X; Expr.ExprSetProperty(A.pExpr, EP_xIsSelect); sqlite3ExprSetHeight(pParse, A.pExpr); }
	else sqlite3SelectDelete(pParse.db, ref X);
    A.zStart = B.z; A.zEnd = E.z.Substring(E.n); }
expr(A) ::= expr(X) in_op(N) LP select(Y) RP(E). [IN] {
	A.pExpr = sqlite3PExpr(pParse, TK.IN, X.pExpr, 0, 0);
    if (A.pExpr != null) { A.pExpr.x.pSelect = Y; Expr.ExprSetProperty(A.pExpr, EP_xIsSelect); sqlite3ExprSetHeight(pParse, A.pExpr);
    else sqlite3SelectDelete(pParse.db, ref Y);
    if (N != 0) A.pExpr = sqlite3PExpr(pParse, TK.NOT, A.pExpr, 0, 0);
    A.zStart = X.zStart; A.zEnd = E.z.Substring(E.n);
}
expr(A) ::= expr(X) in_op(N) nm(Y) dbnm(Z). [IN] {
	var pSrc = sqlite3SrcListAppend(pParse.db, 0, Y, Z);
    A.pExpr = sqlite3PExpr(pParse, TK.IN, X.pExpr, 0, 0);
    if (A.pExpr != null) { A.pExpr.x.pSelect = sqlite3SelectNew(pParse, 0,pSrc,0,0,0,0,0,0,0); Expr.ExprSetProperty(A.pExpr, EP_xIsSelect); sqlite3ExprSetHeight(pParse, A.pExpr); }
	else sqlite3SrcListDelete(pParse.db, ref pSrc);
    if (N != 0) A.pExpr = sqlite3PExpr(pParse, TK.NOT, A.pExpr, 0, 0);
    A.zStart = X.zStart; A.zEnd = (Z.z != null ? Z.z.Substring(Z.n) : Y.z.Substring(Y.n));
}
expr(A) ::= EXISTS(B) LP select(Y) RP(E). {
    var p = A.pExpr = sqlite3PExpr(pParse, TK.EXISTS, 0, 0, 0);
    if (p != null) { p.x.pSelect = Y; Expr.ExprSetProperty(p, EP_xIsSelect); sqlite3ExprSetHeight(pParse, p); }
	else sqlite3SelectDelete(pParse.db, ref Y);
    A.zStart = B.z; A.zEnd = E.z.Substring(E.n);
}
// CASE expressions
expr(A) ::= CASE(C) case_operand(X) case_exprlist(Y) case_else(Z) END(E). {
	A.pExpr = sqlite3PExpr(pParse, TK.CASE, X, Z, 0);
	if (A.pExpr != null) { A.pExpr.x.pList = Y; sqlite3ExprSetHeight(pParse, A.pExpr); }
	else sqlite3ExprListDelete(pParse.db, Y);
	A.zStart = C.z; A.zEnd = E.z.Substring(E.n);
}
%type case_exprlist {ExprList}
%destructor case_exprlist { sqlite3ExprListDelete(pParse.db, $$); }
case_exprlist(A) ::= case_exprlist(X) WHEN expr(Y) THEN expr(Z).	{ A = sqlite3ExprListAppend(pParse, X, Y.pExpr); A = sqlite3ExprListAppend(pParse, A, Z.pExpr); }
case_exprlist(A) ::= WHEN expr(Y) THEN expr(Z).						{ A = sqlite3ExprListAppend(pParse, 0, Y.pExpr); A = sqlite3ExprListAppend(pParse, A, Z.pExpr); }
%type case_else {Expr}
%destructor case_else { sqlite3ExprDelete(pParse.db, $$); }
case_else(A) ::= ELSE expr(X).	{ A = X.pExpr; }
case_else(A) ::= .				{ A = 0; }
%type case_operand {Expr}
%destructor case_operand { sqlite3ExprDelete(pParse.db, $$); }
case_operand(A) ::= expr(X).    { A = X.pExpr; } 
case_operand(A) ::= .           { A = 0; } 
//
%type exprlist {ExprList}
%destructor exprlist { sqlite3ExprListDelete(pParse.db, $$); }
%type nexprlist {ExprList}
%destructor nexprlist { sqlite3ExprListDelete(pParse.db, $$); }
exprlist(A) ::= nexprlist(X).					{ A = X; }
exprlist(A) ::= .								{ A = 0; }
nexprlist(A) ::= nexprlist(X) COMMA expr(Y).	{ A = sqlite3ExprListAppend(pParse, X, Y.pExpr); }
nexprlist(A) ::= expr(Y).						{ A = sqlite3ExprListAppend(pParse, 0, Y.pExpr); }


///////////////////////////// The PRAGMA command /////////////////////////////
//
cmd ::= PRAGMA nm(X) dbnm(Z).					{ sqlite3Pragma(pParse, X, Z, 0, 0); }
cmd ::= PRAGMA nm(X) dbnm(Z) EQ nmnum(Y).		{ sqlite3Pragma(pParse, X, Z, Y, 0); }
cmd ::= PRAGMA nm(X) dbnm(Z) LP nmnum(Y) RP.	{ sqlite3Pragma(pParse, X, Z, Y, 0); }
cmd ::= PRAGMA nm(X) dbnm(Z) EQ minus_num(Y).	{ sqlite3Pragma(pParse, X, Z, Y, 1); }
cmd ::= PRAGMA nm(X) dbnm(Z) LP minus_num(Y) RP.{ sqlite3Pragma(pParse, X, Z, Y, 1); }
nmnum(A) ::= plus_num(X).			{ A = X; }
nmnum(A) ::= nm(X).					{ A = X; }
nmnum(A) ::= ON(X).					{ A = X; }
nmnum(A) ::= DELETE(X).				{ A = X; }
nmnum(A) ::= DEFAULT(X).			{ A = X; }
plus_num(A) ::= plus_opt number(X).	{ A = X; }
minus_num(A) ::= MINUS number(X).	{ A = X; }
number(A) ::= INTEGER|FLOAT(X).		{ A = X; }
plus_opt ::= PLUS.
plus_opt ::= .


//////////////////////// ALTER TABLE table ... ////////////////////////////////
//
cmd ::= ALTER TABLE fullname(X) RENAME TO nm(Z).					{ sqlite3AlterRenameTable(pParse, X, Z); }
cmd ::= ALTER TABLE add_column_fullname ADD kwcolumn_opt column(Y). { sqlite3AlterFinishAddColumn(pParse, Y); }
add_column_fullname ::= fullname(X). { pParse.db.lookaside.bEnabled = 0; sqlite3AlterBeginAddColumn(pParse, X); }
kwcolumn_opt ::= .
kwcolumn_opt ::= COLUMNKW.