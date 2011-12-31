using System;
using System.Collections.Generic;

namespace Lemon
{
    public class Config
    {
        public static ConfigCollection Configs = new ConfigCollection();
        public Rule Rule;
        public int Dot;
        public HashSet<int> fws = new HashSet<int>();
        public LinkedList<Config> fplp;
        public LinkedList<Config> bplp;
        public State State;
        public bool Complete;
        public Config Next;
        public Config Prev;
    }
}
