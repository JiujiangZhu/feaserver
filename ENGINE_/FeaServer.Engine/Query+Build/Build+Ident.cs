using System.Text;
namespace FeaServer.Engine.Query
{
    public partial class Build
    {
        // Measure the number of characters needed to output the given identifier.  The number returned includes any quotes used
        // but does not include the null terminator.
        //
        // The estimate is conservative.  It might be larger that what is really needed.
        internal static int identLength(string z)
        {
            int n;
            for (n = 0; n < z.Length; n++)
                if (z[n] == (byte)'"')
                    n++;
            return n + 2;
        }

        // The first parameter is a pointer to an output buffer. The second parameter is a pointer to an integer that contains the offset at
        // which to write into the output buffer. This function copies the nul-terminated string pointed to by the third parameter, zSignedIdent,
        // to the specified offset in the buffer and updates *pIdx to refer to the first byte after the last byte written before returning.
        // If the string zSignedIdent consists entirely of alpha-numeric characters, does not begin with a digit and is not an SQL keyword,
        // then it is copied to the output buffer exactly as it is. Otherwise, it is quoted using double-quotes.
        internal static void identPut(StringBuilder z, ref int pIdx, string zSignedIdent)
        {
            var zIdent = zSignedIdent;
            var i = pIdx;
            int j;
            for (j = 0; j < zIdent.Length; j++)
                if (!sqlite3Isalnum(zIdent[j]) && zIdent[j] != '_')
                    break;
            var needQuote = (sqlite3Isdigit(zIdent[0]) || sqlite3KeywordCode(zIdent, j) != Parser.TK.ID);
            if (!needQuote)
                needQuote = (j < zIdent.Length && zIdent[j] != 0);
            if (needQuote)
            {
                if (i == z.Length)
                    z.Append('\0');
                z[i++] = '"';
            }
            for (j = 0; j < zIdent.Length; j++)
            {
                if (i == z.Length)
                    z.Append('\0');
                z[i++] = zIdent[j];
                if (zIdent[j] == '"')
                {
                    if (i == z.Length)
                        z.Append('\0');
                    z[i++] = '"';
                }
            }
            if (needQuote)
            {
                if (i == z.Length)
                    z.Append('\0');
                z[i++] = '"';
            }
            pIdx = i;
        }
    }
}