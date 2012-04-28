#region License
/*
The MIT License

Copyright (c) 2009 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#endregion
using System;
using FeaServer.Engine.System;
namespace FeaServer.Engine.Core
{
    public class PCacheGlobal
    {
        static PCacheGlobal pcache = new PCacheGlobal();

        public PGroup grp;                  // The global PGroup for mode (2)
        // Variables related to SQLITE_CONFIG_PAGECACHE settings.  The szSlot, nSlot, pStart, pEnd, nReserve, and isInit values are all
        // fixed at sqlite3_initialize() time and do not require mutex protection. The nFreeSlot and pFree values do require mutex protection.
        public bool isInit;                 // True if initialized
        public int szSlot;                  // Size of each free slot
        public int nSlot;                   // The number of pcache slots
        public int nReserve;                // Try to keep nFreeSlot above this
        public object pStart, pEnd;         // Bounds of pagecache malloc range
        // Above requires no mutex.  Use mutex below for variable that follow.
        public sqlite3_mutex mutex;         // Mutex for accessing the following:
        public int nFreeSlot;               // Number of unused pcache slots
        public PgFreeslot pFree;            // Free page blocks
        // The following value requires a mutex to change.  We skip the mutex on reading because (1) most platforms read a 32-bit integer atomically and
        // (2) even if an incorrect value is read, no great harm is done since this is really just an optimization.
        public bool bUnderPressure;         // True if low on PAGECACHE memory

        public PCacheGlobal()
        {
            grp = new PGroup();
        }
    }
}
