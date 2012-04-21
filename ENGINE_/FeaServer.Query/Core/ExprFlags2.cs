using System;
namespace FeaServer.Core
{
    [Flags]
    public enum ExprFlags2 : byte
    {
        EP2_MallocedToken = 0x0001, /* Need to sqlite3DbFree() Expr.zToken */
        EP2_Irreducible = 0x0002,   /* Cannot EXPRDUP_REDUCE this Expr */
    }
}

///*
//** The pseudo-routine sqlite3ExprSetIrreducible sets the EP2_Irreducible flag on an expression structure.  This flag is used for VV&A only.  The
//** routine is implemented as a macro that only works when in debugging mode, so as not to burden production code.
//*/
//#ifdef SQLITE_DEBUG
//# define ExprSetIrreducible(X)  (X)->flags2 |= EP2_Irreducible
//#else
//# define ExprSetIrreducible(X)
//#endif

///*
//** These macros can be used to test, set, or clear bits in the 
//** Expr.flags field.
//*/
//#define ExprHasProperty(E,P)     (((E)->flags&(P))==(P))
//#define ExprHasAnyProperty(E,P)  (((E)->flags&(P))!=0)
//#define ExprSetProperty(E,P)     (E)->flags|=(P)
//#define ExprClearProperty(E,P)   (E)->flags&=~(P)

///*
//** Macros to determine the number of bytes required by a normal Expr 
//** struct, an Expr struct with the EP_Reduced flag set in Expr.flags 
//** and an Expr struct with the EP_TokenOnly flag set.
//*/
//#define EXPR_FULLSIZE           sizeof(Expr)           /* Full size */
//#define EXPR_REDUCEDSIZE        offsetof(Expr,iTable)  /* Common features */
//#define EXPR_TOKENONLYSIZE      offsetof(Expr,pLeft)   /* Fewer features */

///*
//** Flags passed to the sqlite3ExprDup() function. See the header comment 
//** above sqlite3ExprDup() for details.
//*/
//#define EXPRDUP_REDUCE         0x0001  /* Used reduced-size Expr nodes */
