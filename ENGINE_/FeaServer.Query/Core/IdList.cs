namespace FeaServer.Core
{
    /*
    ** An instance of this structure can hold a simple list of identifiers, such as the list "a,b,c" in the following statements:
    **
    **      INSERT INTO t(a,b,c) VALUES ...;
    **      CREATE INDEX idx ON t(a,b,c);
    **      CREATE TRIGGER trig BEFORE UPDATE ON t(a,b,c) ...;
    **
    ** The IdList.a.idx field is used when the IdList represents the list of column names after a table name in an INSERT statement.  In the statement
    **
    **     INSERT INTO t(a,b,c) ...
    **
    ** If "a" is the k-th column of table "t", then IdList.a[0].idx==k.
    */
    public struct IdList
    {
        struct IdList_item
        {
            string Name;
            int idx;
        }
        IdList_item a;
        int nId;
    }
}


