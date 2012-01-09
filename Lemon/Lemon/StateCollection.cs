using System;
using System.Collections.Generic;

namespace Lemon
{
    public class StateCollection : Dictionary<Config, State>
    {
        public State GetState(Context ctx)
        {
            var basis = Config.Configs.GetBasis();
            State state;
            if (base.TryGetValue(basis, out state))
            {
                /* Copy all the follow-set propagation links from the state under construction into the preexisting state, then return a pointer to the preexisting state */
                for (Config x = basis, y = state.Basis; x != null && y != null; x = x.Prev, y = y.Prev)
                {
                    foreach (var item in x.Basis)
                        y.Basis.AddFirst(item);
                    x.Forwards.Clear();
                    x.Forwards = x.Basis = null;
                }
                var config = Config.Configs.GetItem();
                Config.Configs.Eat(config);
            }
            else
            {
                /* This really is a new state.  Construct all the details */
                Config.Configs.Closure(ctx);
                var config = Config.Configs.GetItem();
                state = new State
                {
                    Basis = basis,
                    Config = config,
                    ID = ctx.States++,
                };
                State.States.Add(state.Basis, state);
                /* Recursively compute successor states */
                ctx.BuildShifts(state);
            }
            return state;
        }
    }
}
