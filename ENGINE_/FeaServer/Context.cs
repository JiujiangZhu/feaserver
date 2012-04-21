using FeaServer.Auth;
namespace FeaServer
{
    public interface IMutex
    {
        void Enter();
        void Leave();
    }

    public class Database
    {
        public string Name;
    }

    public class Context
    {
        public IMutex Mutex;
        public Authorizer Authorizer;
        public object AuthorizerArg;
        public Database[] aDb;
        public int nDb;
    }
}
