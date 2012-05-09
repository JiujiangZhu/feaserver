using System;
namespace Contoso.Sys
{
    [Flags]
    public enum SQLITE
    {
        OK = 0,
        ERROR = 1,
        INTERNAL = 2,
        PERM = 3,
        ABORT = 4,
        BUSY = 5,
        LOCKED = 6,
        NOMEM = 7,
        READONLY = 8,
        INTERRUPT = 9,
        IOERR = 10,
        CORRUPT = 11,
        NOTFOUND = 12,
        FULL = 13,
        CANTOPEN = 14,
        PROTOCOL = 15,
        EMPTY = 16,
        SCHEMA = 17,
        TOOBIG = 18,
        CONSTRAINT = 19,
        MISMATCH = 20,
        MISUSE = 21,
        NOLFS = 22,
        AUTH = 23,
        FORMAT = 24,
        RANGE = 25,
        NOTADB = 26,
        ROW = 100,
        DONE = 101,
    }
}
