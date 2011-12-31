using System;
using System.IO;

namespace Lemon
{
    public class Template
    {
        public static string user_templatename { get; set; }

        /* The next cluster of routines are for reading the template file and writing the results to the generated parser */
        /* The first function transfers data from "in" to "out" until a line is seen which begins with "%%".  The line number is tracked.
        * if name!=0, then any word that begin with "Parse" is changed to begin with *name instead. */
        internal static void tplt_xfer(string name, StreamReader r, StreamWriter w, ref int lineno)
        {
            string line;
            while ((line = r.ReadLine()) != null && (line[0] != '%' || line[1] != '%'))
            {
                lineno++;
                var iStart = 0;
                if (name != null)
                    for (var i = 0; i < line.Length; i++)
                        if (line[i] == 'P' && line.Substring(i, 5) == "Parse" && (i == 0 || !char.IsLetter(line[i - 1])))
                        {
                            if (i > iStart) w.Write(line.Substring(iStart, i - iStart));
                            w.Write(name);
                            i += 4;
                            iStart = i + 1;
                        }
                w.Write(line.Substring(iStart));
            }
        }

        /* The next function finds the template file and opens it, returning a pointer to the opened file. */
        internal static StreamReader tplt_open(Context ctx)
        {
            const string _templateName = "lempar.c";
            FileInfo fileInfo;
            FileStream r;
            /* first, see if user specified a template filename on the command line. */
            if (user_templatename != null)
            {
                try { fileInfo = new FileInfo(user_templatename); }
                catch { fileInfo = null; }
                if (fileInfo == null || fileInfo.IsReadOnly)
                {
                    Console.WriteLine("Can't find the parser driver template file \"{0}\".", user_templatename);
                    ctx.Errors++;
                    return null;
                }
                try { r = fileInfo.Open(FileMode.Open); }
                catch
                {
                    Console.WriteLine("Can't open the template file \"{0}\".", user_templatename);
                    ctx.Errors++;
                    return null;
                }
                return new StreamReader(r);
            }
            var templateName = Path.GetFileNameWithoutExtension(ctx.Filename) + ".lt";
            try { fileInfo = new FileInfo(templateName); }
            catch { fileInfo = null; }
            if (fileInfo == null || fileInfo.IsReadOnly)
            {
                templateName = _templateName;
                try { fileInfo = new FileInfo(templateName); }
                catch { fileInfo = null; }
            }
            if (fileInfo == null)
            {
                Console.WriteLine("Can't find the parser driver template file \"{0}\".", _templateName);
                ctx.Errors++;
                return null;
            }
            try { r = fileInfo.Open(FileMode.Open); }
            catch
            {
                Console.WriteLine("Can't open the template file \"{0}\".", _templateName);
                ctx.Errors++;
                return null;
            }
            return new StreamReader(r);
        }

        /* Print a #line directive line to the output file. */
        internal static void tplt_linedir(StreamWriter w, int lineno, string filename)
        {
            w.WriteLine(string.Format("#line {0} \"{1}\"", lineno, filename.Replace("\\", "\\\\")));
        }

        /* Print a string to the file and keep the linenumber up to date */
        internal static void tplt_print(StreamWriter w, Context ctx, string value, ref int lineno)
        {
            if (value == null) return;
            w.Write(value);
            lineno += (value.Length - value.Replace("\n", string.Empty).Length);
            if (!value.EndsWith("\n"))
                w.WriteLine(ref lineno);
            if (!ctx.NoShowLinenos) { lineno++; tplt_linedir(w, lineno, ctx.Outname); }
        }
    }
}
