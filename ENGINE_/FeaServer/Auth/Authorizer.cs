namespace FeaServer.Auth
{
    public delegate int Authorizer(object arg, AuthorizerAC ac, string table, string column, string db, string authContext);
}
