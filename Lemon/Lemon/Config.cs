using System;
using System.Collections.Generic;

namespace Lemon
{
    public class Config
    {
        public Rule Rule;
        public int Dot;
        public HashSet<int> FwSet;
        public List<Config> Forwards;
        public List<Config> Basises;
        public State State;
        public bool Complete;

        public class KeyComparer : IComparer<Config>
        {
            public int Compare(Config x, Config y)
            {
                var v = x.Rule.ID - y.Rule.ID;
                if (v == 0) v = x.Dot - y.Dot;
                return v;
            }
        }

        public class KeyEqualityComparer : IEqualityComparer<Config>
        {
            public bool Equals(Config x, Config y)
            {
                if (x == null && y == null) return true;
                if (x == null || y == null) return false;
                return (x.Rule == y.Rule && x.Dot == y.Dot);
            }

            public int GetHashCode(Config obj)
            {
                if (obj == null)
                    return 0;
                return obj.Rule.GetHashCode() ^ obj.Dot.GetHashCode();
            }
        }

        public class ListKeyEqualityComparer : IEqualityComparer<List<Config>>
        {
            public bool Equals(List<Config> x, List<Config> y)
            {
                if (x == null && y == null) return true;
                if (x == null || y == null) return false;
                if (x.Count != y.Count) return false;
                using (var xE = x.GetEnumerator())
                using (var yE = y.GetEnumerator())
                    while (xE.MoveNext() && yE.MoveNext())
                    {
                        var x2 = xE.Current;
                        var y2 = yE.Current;
                        if (x2.Rule.ID != y2.Rule.ID || x2.Dot != y2.Dot)
                            return false;
                    }
                return true;
            }

            public int GetHashCode(List<Config> obj)
            {
                if (obj == null)
                    return 0;
                var v = 0;
                foreach (var config in obj)
                    v = (v * 571) + (config.Rule.ID * 37) + config.Dot;
                return v;
            }
        }
    }
}
