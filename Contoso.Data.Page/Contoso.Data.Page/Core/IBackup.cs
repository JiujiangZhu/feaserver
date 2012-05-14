using Pgno = System.UInt32;
namespace Contoso.Core
{
    public interface IBackup
    {
        void sqlite3BackupUpdate(Pgno pgno, byte[] aData);
        void sqlite3BackupRestart();
    }
}
