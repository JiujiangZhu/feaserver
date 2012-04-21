namespace FeaServer.Auth
{
    public enum AuthorizerAC
    {
        CREATE_INDEX,       /* Index Name      Table Name      */
        CREATE_TABLE,       /* Table Name      NULL            */
        CREATE_TEMP_INDEX,  /* Index Name      Table Name      */
        CREATE_TEMP_TABLE,  /* Table Name      NULL            */
        CREATE_TEMP_TRIGGER,/* Trigger Name    Table Name      */
        CREATE_TEMP_VIEW,   /* View Name       NULL            */
        CREATE_TRIGGER,     /* Trigger Name    Table Name      */
        CREATE_VIEW,        /* View Name       NULL            */
        DELETE,             /* Table Name      NULL            */
        DROP_INDEX,         /* Index Name      Table Name      */
        DROP_TABLE,         /* Table Name      NULL            */
        DROP_TEMP_INDEX,    /* Index Name      Table Name      */
        DROP_TEMP_TABLE,    /* Table Name      NULL            */
        DROP_TEMP_TRIGGER,  /* Trigger Name    Table Name      */
        DROP_TEMP_VIEW,     /* View Name       NULL            */
        DROP_TRIGGER,       /* Trigger Name    Table Name      */
        DROP_VIEW,          /* View Name       NULL            */
        INSERT,             /* Table Name      NULL            */
        PRAGMA,             /* Pragma Name     1st arg or NULL */
        READ,               /* Table Name      Column Name     */
        SELECT,             /* NULL            NULL            */
        TRANSACTION,        /* Operation       NULL            */
        UPDATE,             /* Table Name      Column Name     */
        ATTACH,             /* Filename        NULL            */
        DETACH,             /* Database Name   NULL            */
        ALTER_TABLE,        /* Database Name   Table Name      */
        REINDEX,            /* Index Name      NULL            */
        ANALYZE,            /* Table Name      NULL            */
        CREATE_VTABLE,      /* Table Name      Module Name     */
        DROP_VTABLE,        /* Table Name      Module Name     */
        FUNCTION,           /* NULL            Function Name   */
        SAVEPOINT,          /* Operation       Savepoint Name  */
    }
}
