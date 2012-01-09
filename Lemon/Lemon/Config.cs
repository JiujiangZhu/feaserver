using System;
using System.Collections.Generic;

namespace Lemon
{
    public class Config
    {
        public static ConfigCollection Configs = new ConfigCollection();
        public Rule Rule;
        public int Dot;
        public HashSet<int> FwSet = new HashSet<int>();
        public LinkedList<Config> Forwards;
        public LinkedList<Config> Basis;
        public State State;
        public bool Complete;
        public Config Next;
        public Config Prev;

        public class KeyComparer : IEqualityComparer<Config>
        {
            public bool Equals(Config x, Config y)
            {
                if (x == null && y == null) return true;
                if (x == null || y == null) return false;
                return (x.Rule == y.Rule && x.Dot == y.Dot);
            }

            public int GetHashCode(Config obj) { return (obj != null ? obj.Rule.GetHashCode() ^ obj.Dot.GetHashCode() : 0); }
        }
    }
}
