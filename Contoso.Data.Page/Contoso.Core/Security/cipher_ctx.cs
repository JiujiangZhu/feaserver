using System.Security.Cryptography;
namespace Contoso.Security
{
#if SQLITE_HAS_CODEC
    public class cipher_ctx
    {
        public string pass;
        public int pass_sz;
        public bool derive_key;
        public byte[] key;
        public int key_sz;
        public byte[] iv;
        public int iv_sz;
        public ICryptoTransform encryptor;
        public ICryptoTransform decryptor;

        public cipher_ctx Copy()
        {
            cipher_ctx c = new cipher_ctx();
            c.derive_key = derive_key;
            c.pass = pass;
            c.pass_sz = pass_sz;
            if (key != null)
            {
                c.key = new byte[key.Length];
                key.CopyTo(c.key, 0);
            }
            c.key_sz = key_sz;
            if (iv != null)
            {
                c.iv = new byte[iv.Length];
                iv.CopyTo(c.iv, 0);
            }
            c.iv_sz = iv_sz;
            c.encryptor = encryptor;
            c.decryptor = decryptor;
            return c;
        }

        public void CopyTo(cipher_ctx ct)
        {
            ct.derive_key = derive_key;
            ct.pass = pass;
            ct.pass_sz = pass_sz;
            if (key != null)
            {
                ct.key = new byte[key.Length];
                key.CopyTo(ct.key, 0);
            }
            ct.key_sz = key_sz;
            if (iv != null)
            {
                ct.iv = new byte[iv.Length];
                iv.CopyTo(ct.iv, 0);
            }
            ct.iv_sz = iv_sz;
            ct.encryptor = encryptor;
            ct.decryptor = decryptor;
        }
    }
#endif

}
