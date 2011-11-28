using System;
using System.IO;

namespace Lemon
{
    public class Reporter
    {

void ConfigPrint(FILE *fp, struct config *cfp)
{
  struct rule *rp;
  struct symbol *sp;
  int i, j;
  rp = cfp->rp;
  fprintf(fp,"%s ::=",rp->lhs->name);
  for(i=0; i<=rp->nrhs; i++){
    if( i==cfp->dot ) fprintf(fp," *");
    if( i==rp->nrhs ) break;
    sp = rp->rhs[i];
    fprintf(fp," %s", sp->name);
    if( sp->type==MULTITERMINAL ){
      for(j=1; j<sp->nsubsym; j++){
        fprintf(fp,"|%s",sp->subsym[j]->name);
      }
    }
  }
}

/* #define TEST */
#if 0
/* Print a set */
PRIVATE void SetPrint(out,set,lemp)
FILE *out;
char *set;
struct lemon *lemp;
{
  int i;
  char *spacer;
  spacer = "";
  fprintf(out,"%12s[","");
  for(i=0; i<lemp->nterminal; i++){
    if( SetFind(set,i) ){
      fprintf(out,"%s%s",spacer,lemp->symbols[i]->name);
      spacer = " ";
    }
  }
  fprintf(out,"]\n");
}

/* Print a plink chain */
PRIVATE void PlinkPrint(out,plp,tag)
FILE *out;
struct plink *plp;
char *tag;
{
  while( plp ){
    fprintf(out,"%12s%s (state %2d) ","",tag,plp->cfp->stp->statenum);
    ConfigPrint(out,plp->cfp);
    fprintf(out,"\n");
    plp = plp->next;
  }
}
#endif

/* Print an action to the given file descriptor.  Return FALSE if
** nothing was actually printed.
*/
int PrintAction(struct action *ap, FILE *fp, int indent){
  int result = 1;
  switch( ap->type ){
    case SHIFT:
      fprintf(fp,"%*s shift  %d",indent,ap->sp->name,ap->x.stp->statenum);
      break;
    case REDUCE:
      fprintf(fp,"%*s reduce %d",indent,ap->sp->name,ap->x.rp->index);
      break;
    case ACCEPT:
      fprintf(fp,"%*s accept",indent,ap->sp->name);
      break;
    case ERROR:
      fprintf(fp,"%*s error",indent,ap->sp->name);
      break;
    case SRCONFLICT:
    case RRCONFLICT:
      fprintf(fp,"%*s reduce %-3d ** Parsing conflict **",
        indent,ap->sp->name,ap->x.rp->index);
      break;
    case SSCONFLICT:
      fprintf(fp,"%*s shift  %-3d ** Parsing conflict **", 
        indent,ap->sp->name,ap->x.stp->statenum);
      break;
    case SH_RESOLVED:
      if( showPrecedenceConflict ){
        fprintf(fp,"%*s shift  %-3d -- dropped by precedence",
                indent,ap->sp->name,ap->x.stp->statenum);
      }else{
        result = 0;
      }
      break;
    case RD_RESOLVED:
      if( showPrecedenceConflict ){
        fprintf(fp,"%*s reduce %-3d -- dropped by precedence",
                indent,ap->sp->name,ap->x.rp->index);
      }else{
        result = 0;
      }
      break;
    case NOT_USED:
      result = 0;
      break;
  }
  return result;
}

/* Generate the "y.output" log file */
void ReportOutput(struct lemon *lemp)
{
  int i;
  struct state *stp;
  struct config *cfp;
  struct action *ap;
  FILE *fp;

  fp = file_open(lemp,".out","wb");
  if( fp==0 ) return;
  for(i=0; i<lemp->nstate; i++){
    stp = lemp->sorted[i];
    fprintf(fp,"State %d:\n",stp->statenum);
    if( lemp->basisflag ) cfp=stp->bp;
    else                  cfp=stp->cfp;
    while( cfp ){
      char buf[20];
      if( cfp->dot==cfp->rp->nrhs ){
        sprintf(buf,"(%d)",cfp->rp->index);
        fprintf(fp,"    %5s ",buf);
      }else{
        fprintf(fp,"          ");
      }
      ConfigPrint(fp,cfp);
      fprintf(fp,"\n");
#if 0
      SetPrint(fp,cfp->fws,lemp);
      PlinkPrint(fp,cfp->fplp,"To  ");
      PlinkPrint(fp,cfp->bplp,"From");
#endif
      if( lemp->basisflag ) cfp=cfp->bp;
      else                  cfp=cfp->next;
    }
    fprintf(fp,"\n");
    for(ap=stp->ap; ap; ap=ap->next){
      if( PrintAction(ap,fp,30) ) fprintf(fp,"\n");
    }
    fprintf(fp,"\n");
  }
  fprintf(fp, "----------------------------------------------------\n");
  fprintf(fp, "Symbols:\n");
  for(i=0; i<lemp->nsymbol; i++){
    int j;
    struct symbol *sp;

    sp = lemp->symbols[i];
    fprintf(fp, "  %3d: %s", i, sp->name);
    if( sp->type==NONTERMINAL ){
      fprintf(fp, ":");
      if( sp->lambda ){
        fprintf(fp, " <lambda>");
      }
      for(j=0; j<lemp->nterminal; j++){
        if( sp->firstset && SetFind(sp->firstset, j) ){
          fprintf(fp, " %s", lemp->symbols[j]->name);
        }
      }
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
  return;
}

    }
}
