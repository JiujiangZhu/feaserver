using System;
using System.Collections.Generic;

namespace Lemon
{
    public class Config
    {
        public static ConfigCollection Configs = new ConfigCollection();

        public Rule rp;
        public int dot;
        public HashSet<int> fws = new HashSet<int>();
        public LinkedList<Config> fplp;
        public LinkedList<Config> bplp;
        public State stp;
        public bool Complete;
        public Config next;
        public Config bp;
    }
}
