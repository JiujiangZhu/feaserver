using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Lemon
{
    public class State
    {
        public const int NO_OFFSET = int.MaxValue;

        public int ID;
        public List<Config> Basises = new List<Config>();
        public List<Config> Configs = new List<Config>();
        public List<Action> Actions = new List<Action>();
        internal int TokenActions;
        internal int NonTerminalActions;
        internal int TokenOffset;
        internal int NonTerminalOffset;
        internal int Default;

        public class KeyComparer : IComparer<State>
        {
            public int Compare(State x, State y)
            {
                if (object.ReferenceEquals(x, y))
                    return 0;
                var v = y.NonTerminalActions - x.NonTerminalActions;
                if (v == 0)
                {
                    v = y.TokenActions - x.TokenActions;
                    if (v == 0)
                        v = y.ID - x.ID;
                }
                Debug.Assert(v != 0);
                return v;
            }
        }
    }
}
