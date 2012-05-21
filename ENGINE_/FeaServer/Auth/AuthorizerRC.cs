namespace FeaServer.Auth
{
    public enum AuthorizerRC : int
    {
        DENY = 1,   // Abort the SQL statement with an error
        IGNORE = 2, // Don't allow access, but don't generate an error
    }
}
