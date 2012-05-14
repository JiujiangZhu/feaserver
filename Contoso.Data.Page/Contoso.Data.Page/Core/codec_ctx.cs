using System.Security.Cryptography;
namespace Contoso.Core
{
#if SQLITE_HAS_CODEC
    public class codec_ctx
    {
        internal const int ENCRYPT_WRITE_CTX = 6; // Encode page 
        internal const int ENCRYPT_READ_CTX = 7; // Encode page 
        internal const int DECRYPT = 3; // Decode page
        internal const int FILE_HEADER_SZ = 16;
        internal const string CIPHER = "aes-256-cbc";
        internal const int CIPHER_DECRYPT = 0;
        internal const int CIPHER_ENCRYPT = 1;

        public delegate byte[] dxCodec(codec_ctx pCodec, byte[] D, uint pageNumber, int X);
        public delegate void dxCodecSizeChng(codec_ctx pCodec, int pageSize, short nReserve);
        public delegate void dxCodecFree(ref codec_ctx pCodec);

#if NET_2_0
        internal static RijndaelManaged Aes = new RijndaelManaged();
#else
        internal static AesManaged Aes = new AesManaged();
#endif

        public int mode_rekey;
        public byte[] buffer;
        public Btree pBt;
        public cipher_ctx read_ctx;
        public cipher_ctx write_ctx;

        public codec_ctx Copy()
        {
            codec_ctx c = new codec_ctx();
            c.mode_rekey = mode_rekey;
            c.buffer = new byte[buffer.Length];
            c.pBt = pBt;
            if (read_ctx != null)
                c.read_ctx = read_ctx.Copy();
            if (write_ctx != null)
                c.write_ctx = write_ctx.Copy();
            return c;
        }
    }
#endif
}
