using System;
using System.Collections.Generic;

namespace Lemon
{
    public class StateCollection : Dictionary<Config, State>
    {
        public State GetState(Context ctx)
        {
            /* Extract the sorted basis of the new state.  The basis was constructed by prior calls to "Configlist_addbasis()". */
            var bp = Config.Configs.GetBasisItem();
            /* Get a state with the same basis */
            var stp = base[bp];
            if (stp != null)
            {
                /* A state with the same basis already exists!  Copy all the follow-set propagation links from the state under construction into the preexisting state, then return a pointer to the preexisting state */
                for (Config x = bp, y = stp.bp; (x != null) && (y != null); x = x.Prev, y = y.Prev)
                {
                    foreach (var item in x.bplp)
                        y.bplp.AddFirst(item);
                    x.fplp.Clear();
                    x.fplp = x.bplp = null;
                }
                var cfp = Config.Configs.GetItem();
                Config.Configs.Eat(cfp);
            }
            else
            {
                /* This really is a new state.  Construct all the details */
                Config.Configs.Closure(ctx);   
                var cfp = Config.Configs.GetItem();
                stp = new State
                {
                    bp = bp,
                    cfp = cfp,
                    statenum = ctx.States++,
                };
                State.States.Add(stp.bp, stp);
                /* Recursively compute successor states */
                ctx.BuildShifts(stp);
            }
            return stp;
        }
    }
}
