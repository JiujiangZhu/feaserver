using System;
using Contoso.Core;
using Contoso.Sys;
namespace Contoso
{
    class Program
    {
        static void Main(string[] args)
        {
            var vfs = FileEx.sqlite3_vfs_find(null);
            Pager pager;
            Pager.PAGEROPEN flags = 0;
            VirtualFileSystem.OPEN vfsFlags =  VirtualFileSystem.OPEN.CREATE | VirtualFileSystem.OPEN.READWRITE | VirtualFileSystem.OPEN.MAIN_DB;
            var rc = Pager.sqlite3PagerOpen(vfs, out pager, @"Test", 0, flags, vfsFlags, x => { });
            //PgHdr p = null;
            //pager.sqlite3PagerGet(1, ref p);
            pager.sqlite3PagerClose();
            //VirtualFileSystem.OPEN flagOut = 0;
            //VirtualFile file = new VirtualFile();
            //var rc = FileEx.sqlite3OsOpen(vfs, @"C:\_T\Test", file, VirtualFileSystem.OPEN.CREATE, ref flagOut);
        }
    }
}
