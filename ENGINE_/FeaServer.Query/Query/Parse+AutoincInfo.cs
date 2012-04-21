namespace FeaServer.Query
{
    public partial class Parse
    {
        internal class AutoincInfo
        {
            public AutoincInfo pNext; // Next info block in a list of them all
            public Table pTab;  // Table this info block refers to
            public int iDb;     // Index in sqlite3.aDb[] of database holding pTab
            public int regCtr;  // Memory register holding the rowid counter
        }
    }
}
