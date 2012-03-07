using System;

namespace Lemon
{
    public enum ActionType
    {
        Shift,
        Accept,
        Reduce,
        Error,
        SSConflict,
        SRConflict,
        RRConflict,
        SHResolved,
        RDResolved,
        NotUsed,
    }
}
