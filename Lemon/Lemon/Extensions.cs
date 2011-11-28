using System;
using System.Collections.Generic;

namespace Lemon
{
    public static class Extensions
    {
        public static bool Union<T>(this HashSet<T> source, HashSet<T> set)
        {
            var change = false;
            foreach (var item in source)
                change |= set.Add(item);
            return change;
        }

        //PRIVATE char *pathsearch(char *argv0, char *name, int modemask)
        //{
        //  const char *pathlist;
        //  char *pathbufptr;
        //  char *pathbuf;
        //  char *path,*cp;
        //  char c;

        //#ifdef __WIN32__
        //  cp = strrchr(argv0,'\\');
        //#else
        //  cp = strrchr(argv0,'/');
        //#endif
        //  if( cp ){
        //    c = *cp;
        //    *cp = 0;
        //    path = (char *)malloc( lemonStrlen(argv0) + lemonStrlen(name) + 2 );
        //    if( path ) sprintf(path,"%s/%s",argv0,name);
        //    *cp = c;
        //  }else{
        //    pathlist = getenv("PATH");
        //    if( pathlist==0 ) pathlist = ".:/bin:/usr/bin";
        //    pathbuf = (char *) malloc( lemonStrlen(pathlist) + 1 );
        //    path = (char *)malloc( lemonStrlen(pathlist)+lemonStrlen(name)+2 );
        //    if( (pathbuf != 0) && (path!=0) ){
        //      pathbufptr = pathbuf;
        //      strcpy(pathbuf, pathlist);
        //      while( *pathbuf ){
        //        cp = strchr(pathbuf,':');
        //        if( cp==0 ) cp = &pathbuf[lemonStrlen(pathbuf)];
        //        c = *cp;
        //        *cp = 0;
        //        sprintf(path,"%s/%s",pathbuf,name);
        //        *cp = c;
        //        if( c==0 ) pathbuf[0] = 0;
        //        else pathbuf = &cp[1];
        //        if( access(path,modemask)==0 ) break;
        //      }
        //      free(pathbufptr);
        //    }
        //  }
        //  return path;
        //}

    }
}
