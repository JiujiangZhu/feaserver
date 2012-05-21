using FeaServer.Core;
namespace FeaServer
{
    public class Trigger
    {
        public string Name;       // The name of the trigger
        public string Table;       // The table or view to which the trigger applies
        public byte OP;            // One of TK_DELETE, TK_UPDATE, TK_INSERT
        public byte tr_tm;         // One of TRIGGER_BEFORE, TRIGGER_AFTER
        public Expr When;         // The WHEN clause of the expression (may be NULL)
        public IdList Columns;    // If this is an UPDATE OF <column-list> trigger, the <column-list> is stored here
        public Schema Schema;     // Schema containing the trigger
        public Schema TabSchema;  // Schema containing the table
        //public TriggerStep step_list; // Link list of trigger program steps
        public Trigger Next;      // Next trigger associated with the table
    }
}
