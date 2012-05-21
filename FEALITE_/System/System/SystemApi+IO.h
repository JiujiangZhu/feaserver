#ifndef _SYSTEMAPI_IO_H_
#define _SYSTEMAPI_IO_H_
#ifdef __cplusplus
extern "C" {
#endif

/*
** API: Name Of The Folder Holding Temporary Files
**
** ^(If this global variable is made to point to a string which is the name of a folder (a.k.a. directory), then all temporary files
** created by APPID when using a built-in [system_vfs | VFS] will be placed in that directory.)^  ^If this variable
** is a NULL pointer, then APPID performs a search for an appropriate temporary file directory.
**
** It is not safe to read or modify this variable in more than one thread at a time.  It is not safe to read or modify this variable
** if a [AppContext] is being used at the same time in a separate thread.
** It is intended that this variable be set once as part of process initialization and before any APPID interface
** routines have been called and that this variable remain unchanged thereafter.
**
** ^The [temp_store_directory pragma] may modify this variable and cause it to point to memory obtained from [system_malloc].  ^Furthermore,
** the [temp_store_directory pragma] always assumes that any string that this variable points to is held in memory obtained from 
** [system_malloc] and the pragma may attempt to free that memory using [system_free].
** Hence, if this variable is modified directly, either it should be made NULL or made to point to memory obtained from [system_malloc]
** or else the use of the [temp_store_directory pragma] should be avoided.
*/
SYSTEM_API SYSTEM_EXTERN char *system_temp_directory;

/*
** API: Extended Result Codes
** KEYWORDS: {extended error code} {extended error codes}
** KEYWORDS: {extended result code} {extended result codes}
**
** In its default configuration, APPID API routines return one of 26 integer [SYSTEM_OK | result codes].  However, experience has shown that many of
** these result codes are too coarse-grained.  They do not provide as much information about problems as programmers might like.  In an effort to
** address this, newer versions of APPID (version 3.3.8 and later) include support for additional result codes that provide more detailed information
** about errors. The extended result codes are enabled or disabled on a per database connection basis using the
** [system_extended_result_codes()] API.
**
** Some of the available extended result codes are listed here. One may expect the number of extended result codes will be expand
** over time.  Software that uses extended result codes should expect to see new result codes in future releases of APPID.
**
** The SYSTEM_OK result code will never be extended.  It will always be exactly zero.
*/
#define SYSTEM_IOERR_READ              (SYSTEM_IOERR | (1<<8))
#define SYSTEM_IOERR_SHORT_READ        (SYSTEM_IOERR | (2<<8))
#define SYSTEM_IOERR_WRITE             (SYSTEM_IOERR | (3<<8))
#define SYSTEM_IOERR_FSYNC             (SYSTEM_IOERR | (4<<8))
#define SYSTEM_IOERR_DIR_FSYNC         (SYSTEM_IOERR | (5<<8))
#define SYSTEM_IOERR_TRUNCATE          (SYSTEM_IOERR | (6<<8))
#define SYSTEM_IOERR_FSTAT             (SYSTEM_IOERR | (7<<8))
#define SYSTEM_IOERR_UNLOCK            (SYSTEM_IOERR | (8<<8))
#define SYSTEM_IOERR_RDLOCK            (SYSTEM_IOERR | (9<<8))
#define SYSTEM_IOERR_DELETE            (SYSTEM_IOERR | (10<<8))
#define SYSTEM_IOERR_BLOCKED           (SYSTEM_IOERR | (11<<8))
#define SYSTEM_IOERR_NOMEM             (SYSTEM_IOERR | (12<<8))
#define SYSTEM_IOERR_ACCESS            (SYSTEM_IOERR | (13<<8))
#define SYSTEM_IOERR_CHECKRESERVEDLOCK (SYSTEM_IOERR | (14<<8))
#define SYSTEM_IOERR_LOCK              (SYSTEM_IOERR | (15<<8))
#define SYSTEM_IOERR_CLOSE             (SYSTEM_IOERR | (16<<8))
#define SYSTEM_IOERR_DIR_CLOSE         (SYSTEM_IOERR | (17<<8))
#define SYSTEM_IOERR_SHMOPEN           (SYSTEM_IOERR | (18<<8))
#define SYSTEM_IOERR_SHMSIZE           (SYSTEM_IOERR | (19<<8))
#define SYSTEM_IOERR_SHMLOCK           (SYSTEM_IOERR | (20<<8))
#define SYSTEM_LOCKED_SHAREDCACHE      (SYSTEM_LOCKED |  (1<<8))
#define SYSTEM_BUSY_RECOVERY           (SYSTEM_BUSY   |  (1<<8))
#define SYSTEM_CANTOPEN_NOTEMPDIR      (SYSTEM_CANTOPEN | (1<<8))

/*
** API: Flags For File Open Operations
**
** These bit values are intended for use in the 3rd parameter to the [system_open_v2()] interface and
** in the 4th parameter to the xOpen method of the [system_vfs] object.
*/
#define SYSTEM_OPEN_READONLY         0x00000001  /* Ok for system_open_v2() */
#define SYSTEM_OPEN_READWRITE        0x00000002  /* Ok for system_open_v2() */
#define SYSTEM_OPEN_CREATE           0x00000004  /* Ok for system_open_v2() */
#define SYSTEM_OPEN_DELETEONCLOSE    0x00000008  /* VFS only */
#define SYSTEM_OPEN_EXCLUSIVE        0x00000010  /* VFS only */
#define SYSTEM_OPEN_AUTOPROXY        0x00000020  /* VFS only */
#define SYSTEM_OPEN_MAIN_DB          0x00000100  /* VFS only */
#define SYSTEM_OPEN_TEMP_DB          0x00000200  /* VFS only */
#define SYSTEM_OPEN_TRANSIENT_DB     0x00000400  /* VFS only */
#define SYSTEM_OPEN_MAIN_JOURNAL     0x00000800  /* VFS only */
#define SYSTEM_OPEN_TEMP_JOURNAL     0x00001000  /* VFS only */
#define SYSTEM_OPEN_SUBJOURNAL       0x00002000  /* VFS only */
#define SYSTEM_OPEN_MASTER_JOURNAL   0x00004000  /* VFS only */
#define SYSTEM_OPEN_NOMUTEX          0x00008000  /* Ok for system_open_v2() */
#define SYSTEM_OPEN_FULLMUTEX        0x00010000  /* Ok for system_open_v2() */
#define SYSTEM_OPEN_SHAREDCACHE      0x00020000  /* Ok for system_open_v2() */
#define SYSTEM_OPEN_PRIVATECACHE     0x00040000  /* Ok for system_open_v2() */
#define SYSTEM_OPEN_WAL              0x00080000  /* VFS only */

/*
** API: Device Characteristics
**
** The xDeviceCharacteristics method of the [system_io_methods] object returns an integer which is a vector of the these
** bit values expressing I/O characteristics of the mass storage device that holds the file that the [system_io_methods]
** refers to.
**
** The SYSTEM_IOCAP_ATOMIC property means that all writes of any size are atomic.  The SYSTEM_IOCAP_ATOMICnnn values
** mean that writes of blocks that are nnn bytes in size and are aligned to an address which is an integer multiple of
** nnn are atomic.  The SYSTEM_IOCAP_SAFE_APPEND value means that when data is appended to a file, the data is appended
** first then the size of the file is extended, never the other way around.  The SYSTEM_IOCAP_SEQUENTIAL property means that
** information is written to disk in the same order as calls to xWrite().
*/
#define SYSTEM_IOCAP_ATOMIC                 0x00000001
#define SYSTEM_IOCAP_ATOMIC512              0x00000002
#define SYSTEM_IOCAP_ATOMIC1K               0x00000004
#define SYSTEM_IOCAP_ATOMIC2K               0x00000008
#define SYSTEM_IOCAP_ATOMIC4K               0x00000010
#define SYSTEM_IOCAP_ATOMIC8K               0x00000020
#define SYSTEM_IOCAP_ATOMIC16K              0x00000040
#define SYSTEM_IOCAP_ATOMIC32K              0x00000080
#define SYSTEM_IOCAP_ATOMIC64K              0x00000100
#define SYSTEM_IOCAP_SAFE_APPEND            0x00000200
#define SYSTEM_IOCAP_SEQUENTIAL             0x00000400
#define SYSTEM_IOCAP_UNDELETABLE_WHEN_OPEN  0x00000800

/*
** API: File Locking Levels
**
** APPID uses one of these integer values as the second argument to calls it makes to the xLock() and xUnlock() methods
** of an [system_io_methods] object.
*/
#define SYSTEM_LOCK_NONE          0
#define SYSTEM_LOCK_SHARED        1
#define SYSTEM_LOCK_RESERVED      2
#define SYSTEM_LOCK_PENDING       3
#define SYSTEM_LOCK_EXCLUSIVE     4

/*
** API: Synchronization Type Flags
**
** When APPID invokes the xSync() method of an [system_io_methods] object it uses a combination of
** these integer values as the second argument.
**
** When the SYSTEM_SYNC_DATAONLY flag is used, it means that the sync operation only needs to flush data to mass storage.  Inode
** information need not be flushed. If the lower four bits of the flag equal SYSTEM_SYNC_NORMAL, that means to use normal fsync() semantics.
** If the lower four bits equal SYSTEM_SYNC_FULL, that means to use Mac OS X style fullsync instead of fsync().
*/
#define SYSTEM_SYNC_NORMAL        0x00002
#define SYSTEM_SYNC_FULL          0x00003
#define SYSTEM_SYNC_DATAONLY      0x00010

/*
** API: OS Interface Open File Handle
**
** An [system_file] object represents an open file in the [system_vfs | OS interface layer].  Individual OS interface
** implementations will want to subclass this object by appending additional fields
** for their own use.  The pMethods entry is a pointer to an [system_io_methods] object that defines methods for performing
** I/O operations on the open file.
*/
typedef struct system_file system_file;
struct system_file
{
	const struct system_io_methods *pMethods; // Methods for an open file
};

/*
** API: OS Interface File Virtual Methods Object
**
** Every file opened by the [system_vfs] xOpen method populates an [system_file] object (or, more commonly, a subclass of the
** [system_file] object) with a pointer to an instance of this object. This object defines the methods used to perform various operations
** against the open file represented by the [system_file] object.
**
** If the xOpen method sets the system_file.pMethods element to a non-NULL pointer, then the system_io_methods.xClose method
** may be invoked even if the xOpen reported that it failed.  The only way to prevent a call to xClose following a failed xOpen
** is for the xOpen to set the system_file.pMethods element to NULL.
**
** The flags argument to xSync may be one of [SYSTEM_SYNC_NORMAL] or [SYSTEM_SYNC_FULL].  The first choice is the normal fsync().
** The second choice is a Mac OS X style fullsync.  The [SYSTEM_SYNC_DATAONLY] flag may be ORed in to indicate that only the data of the file
** and not its inode needs to be synced.
**
** The integer values to xLock() and xUnlock() are one of
** <ul>
** <li> [SYSTEM_LOCK_NONE],
** <li> [SYSTEM_LOCK_SHARED],
** <li> [SYSTEM_LOCK_RESERVED],
** <li> [SYSTEM_LOCK_PENDING], or
** <li> [SYSTEM_LOCK_EXCLUSIVE].
** </ul>
** xLock() increases the lock. xUnlock() decreases the lock. The xCheckReservedLock() method checks whether any database connection,
** either in this process or in some other process, is holding a RESERVED, PENDING, or EXCLUSIVE lock on the file.  It returns true
** if such a lock exists and false otherwise.
**
** The xFileControl() method is a generic interface that allows custom VFS implementations to directly control an open file using the
** [system_file_control()] interface.  The second "op" argument is an integer opcode.  The third argument is a generic pointer intended to
** point to a structure that may contain arguments or space in which to write return values.  Potential uses for xFileControl() might be
** functions to enable blocking locks with timeouts, to change the locking strategy (for example to use dot-file locks), to inquire
** about the status of a lock, or to break stale locks.  The APPID core reserves all opcodes less than 100 for its own use.
** A [SYSTEM_FCNTL_LOCKSTATE | list of opcodes] less than 100 is available. Applications that define a custom xFileControl method should use opcodes
** greater than 100 to avoid conflicts.
**
** The xSectorSize() method returns the sector size of the device that underlies the file.  The sector size is the
** minimum write that can be performed without disturbing other bytes in the file.  The xDeviceCharacteristics()
** method returns a bit vector describing behaviors of the underlying device:
**
** <ul>
** <li> [SYSTEM_IOCAP_ATOMIC]
** <li> [SYSTEM_IOCAP_ATOMIC512]
** <li> [SYSTEM_IOCAP_ATOMIC1K]
** <li> [SYSTEM_IOCAP_ATOMIC2K]
** <li> [SYSTEM_IOCAP_ATOMIC4K]
** <li> [SYSTEM_IOCAP_ATOMIC8K]
** <li> [SYSTEM_IOCAP_ATOMIC16K]
** <li> [SYSTEM_IOCAP_ATOMIC32K]
** <li> [SYSTEM_IOCAP_ATOMIC64K]
** <li> [SYSTEM_IOCAP_SAFE_APPEND]
** <li> [SYSTEM_IOCAP_SEQUENTIAL]
** </ul>
**
** The SYSTEM_IOCAP_ATOMIC property means that all writes of any size are atomic.  The SYSTEM_IOCAP_ATOMICnnn values
** mean that writes of blocks that are nnn bytes in size and are aligned to an address which is an integer multiple of
** nnn are atomic.  The SYSTEM_IOCAP_SAFE_APPEND value means that when data is appended to a file, the data is appended
** first then the size of the file is extended, never the other way around.  The SYSTEM_IOCAP_SEQUENTIAL property means that
** information is written to disk in the same order as calls to xWrite().
**
** If xRead() returns SYSTEM_IOERR_SHORT_READ it must also fill in the unread portions of the buffer with zeros.  A VFS that
** fails to zero-fill short reads might seem to work.  However, failure to zero-fill short reads will eventually lead to
** database corruption.
*/
typedef struct system_io_methods system_io_methods;
struct system_io_methods
{
	int iVersion;
	int (*xClose)(system_file*);
	int (*xRead)(system_file*, void*, int iAmt, INT64_TYPE iOfst);
	int (*xWrite)(system_file*, const void*, int iAmt, INT64_TYPE iOfst);
	int (*xTruncate)(system_file*, INT64_TYPE size);
	int (*xSync)(system_file*, int flags);
	int (*xFileSize)(system_file*, INT64_TYPE *pSize);
	int (*xLock)(system_file*, int);
	int (*xUnlock)(system_file*, int);
	int (*xCheckReservedLock)(system_file*, int *pResOut);
	int (*xFileControl)(system_file*, int op, void *pArg);
	int (*xSectorSize)(system_file*);
	int (*xDeviceCharacteristics)(system_file*);
	// Methods above are valid for version 1
	int (*xShmMap)(system_file*, int iPg, int pgsz, int, void volatile**);
	int (*xShmLock)(system_file*, int offset, int n, int flags);
	void (*xShmBarrier)(system_file*);
	int (*xShmUnmap)(system_file*, int deleteFlag);
	// Methods above are valid for version 2
	// Additional methods may be added in future releases
};

/*
** API: Standard File Control Opcodes
**
** These integer constants are opcodes for the xFileControl method of the [system_io_methods] object and for the [system_file_control()]
** interface.
**
** The [SYSTEM_FCNTL_LOCKSTATE] opcode is used for debugging.  This opcode causes the xFileControl method to write the current state of
** the lock (one of [SYSTEM_LOCK_NONE], [SYSTEM_LOCK_SHARED], [SYSTEM_LOCK_RESERVED], [SYSTEM_LOCK_PENDING], or [SYSTEM_LOCK_EXCLUSIVE])
** into an integer that the pArg argument points to. This capability is used during testing and only needs to be supported when SYSTEM_TEST
** is defined.
**
** The [SYSTEM_FCNTL_SIZE_HINT] opcode is used by APPID to give the VFS layer a hint of how large the database file will grow to be during the
** current transaction.  This hint is not guaranteed to be accurate but it is often close.  The underlying VFS might choose to preallocate database
** file space based on this hint in order to help writes to the database file run faster.
**
** The [SYSTEM_FCNTL_CHUNK_SIZE] opcode is used to request that the VFS extends and truncates the database file in chunks of a size specified
** by the user. The fourth argument to [system_file_control()] should point to an integer (type int) containing the new chunk-size to use
** for the nominated database. Allocating database file space in large chunks (say 1MB at a time), may reduce file-system fragmentation and
** improve performance on some systems.
*/
#define SYSTEM_FCNTL_LOCKSTATE        1
#define SYSTEM_GET_LOCKPROXYFILE      2
#define SYSTEM_SET_LOCKPROXYFILE      3
#define SYSTEM_LAST_ERRNO             4
#define SYSTEM_FCNTL_SIZE_HINT        5
#define SYSTEM_FCNTL_CHUNK_SIZE       6

/*
** API: OS Interface Object
**
** An instance of the system_vfs object defines the interface between the APPID core and the underlying operating system.  The "vfs"
** in the name of the object stands for "virtual file system".
**
** The value of the iVersion field is initially 1 but may be larger in future versions of APPID.  Additional fields may be appended to this
** object when the iVersion value is increased.  Note that the structure of the system_vfs object changes in the transaction between
** APPID version 3.5.9 and 3.6.0 and yet the iVersion field was not modified.
**
** The szOsFile field is the size of the subclassed [system_file] structure used by this VFS.  mxPathname is the maximum length of
** a pathname in this VFS.
**
** Registered system_vfs objects are kept on a linked list formed by the pNext pointer.  The [system_vfs_register()]
** and [system_vfs_unregister()] interfaces manage this list in a thread-safe way.  The [system_vfs_find()] interface
** searches the list.  Neither the application code nor the VFS implementation should use the pNext pointer.
**
** The pNext field is the only field in the system_vfs structure that APPID will ever modify.  APPID will only access
** or modify this field while holding a particular static mutex. The application should never modify anything within the system_vfs
** object once the object has been registered.
**
** The zName field holds the name of the VFS module.  The name must be unique across all VFS modules.
**
** ^APPID guarantees that the zFilename parameter to xOpen is either a NULL pointer or string obtained
** from xFullPathname() with an optional suffix added. ^If a suffix is added to the zFilename parameter, it will
** consist of a single "-" character followed by no more than 10 alphanumeric and/or "-" characters.
** ^APPID further guarantees that the string will be valid and unchanged until xClose() is called. Because of the previous sentence,
** the [system_file] can safely store a pointer to the filename if it needs to remember the filename for some reason.
** If the zFilename parameter to xOpen is a NULL pointer then xOpen must invent its own temporary name for the file.  ^Whenever the 
** xFilename parameter is NULL it will also be the case that the flags parameter will include [SYSTEM_OPEN_DELETEONCLOSE].
**
** The flags argument to xOpen() includes all bits set in the flags argument to [system_open_v2()].  Or if [system_open()]
** or [system_open16()] is used, then flags includes at least [SYSTEM_OPEN_READWRITE] | [SYSTEM_OPEN_CREATE]. 
** If xOpen() opens a file read-only then it sets *pOutFlags to include [SYSTEM_OPEN_READONLY].  Other bits in *pOutFlags may be set.
**
** ^(APPID will also add one of the following flags to the xOpen() call, depending on the object being opened:
**
** <ul>
** <li>  [SYSTEM_OPEN_MAIN_DB]
** <li>  [SYSTEM_OPEN_MAIN_JOURNAL]
** <li>  [SYSTEM_OPEN_TEMP_DB]
** <li>  [SYSTEM_OPEN_TEMP_JOURNAL]
** <li>  [SYSTEM_OPEN_TRANSIENT_DB]
** <li>  [SYSTEM_OPEN_SUBJOURNAL]
** <li>  [SYSTEM_OPEN_MASTER_JOURNAL]
** <li>  [SYSTEM_OPEN_WAL]
** </ul>)^
**
** The file I/O implementation can use the object type flags to change the way it deals with files.  For example, an application
** that does not care about crash recovery or rollback might make the open of a journal file a no-op.  Writes to this journal would
** also be no-ops, and any attempt to read the journal would return SYSTEM_IOERR.  Or the implementation might recognize that a database
** file will be doing page-aligned sector reads and writes in a random order and set up its I/O subsystem accordingly.
**
** APPID might also add one of the following flags to the xOpen method:
**
** <ul>
** <li> [SYSTEM_OPEN_DELETEONCLOSE]
** <li> [SYSTEM_OPEN_EXCLUSIVE]
** </ul>
**
** The [SYSTEM_OPEN_DELETEONCLOSE] flag means the file should be deleted when it is closed.  ^The [SYSTEM_OPEN_DELETEONCLOSE]
** will be set for TEMP databases and their journals, transient databases, and subjournals.
**
** ^The [SYSTEM_OPEN_EXCLUSIVE] flag is always used in conjunction with the [SYSTEM_OPEN_CREATE] flag, which are both directly
** analogous to the O_EXCL and O_CREAT flags of the POSIX open() API.  The SYSTEM_OPEN_EXCLUSIVE flag, when paired with the 
** SYSTEM_OPEN_CREATE, is used to indicate that file should always be created, and that it is an error if it already exists.
** It is <i>not</i> used to indicate the file should be opened for exclusive access.
**
** ^At least szOsFile bytes of memory are allocated by APPID to hold the [system_file] structure passed as the third
** argument to xOpen.  The xOpen method does not have to allocate the structure; it should just fill it in.  Note that
** the xOpen method must set the system_file.pMethods to either a valid [system_io_methods] object or to NULL.  xOpen must do
** this even if the open fails.  APPID expects that the system_file.pMethods element will be valid after xOpen returns regardless of the success
** or failure of the xOpen call.
**
** ^The flags argument to xAccess() may be [SYSTEM_ACCESS_EXISTS] to test for the existence of a file, or [SYSTEM_ACCESS_READWRITE] to
** test whether a file is readable and writable, or [SYSTEM_ACCESS_READ] to test whether a file is at least readable.  The file can be a
** directory.
**
** ^APPID will always allocate at least mxPathname+1 bytes for the output buffer xFullPathname.  The exact size of the output buffer
** is also passed as a parameter to both methods. If the output buffer is not large enough, [SYSTEM_CANTOPEN] should be returned. Since this is
** handled as a fatal error by APPID, vfs implementations should endeavor to prevent this by setting mxPathname to a sufficiently large value.
**
** The xRandomness(), xSleep(), xCurrentTime(), and xCurrentTimeInt64() interfaces are not strictly a part of the filesystem, but they are
** included in the VFS structure for completeness. The xRandomness() function attempts to return nBytes bytes
** of good-quality randomness into zOut.  The return value is the actual number of bytes of randomness obtained.
** The xSleep() method causes the calling thread to sleep for at least the number of microseconds given.  ^The xCurrentTime()
** method returns a Julian Day Number for the current date and time as a floating point value.
** ^The xCurrentTimeInt64() method returns, as an integer, the Julian Day Number multipled by 86400000 (the number of milliseconds in a 24-hour day).  
** ^APPID will use the xCurrentTimeInt64() method to get the current date and time if that method is available (if iVersion is 2 or 
** greater and the function pointer is not NULL) and will fall back to xCurrentTime() if xCurrentTimeInt64() is unavailable.
*/
typedef struct system_vfs system_vfs;
struct system_vfs
{
	int iVersion;            /* Structure version number (currently 2) */
	int szOsFile;            /* Size of subclassed system_file */
	int mxPathname;          /* Maximum file pathname length */
	system_vfs *pNext;      /* Next registered VFS */
	const char *zName;       /* Name of this virtual file system */
	void *pAppData;          /* Pointer to application-specific data */
	int (*xOpen)(system_vfs*, const char *zName, system_file*, int flags, int *pOutFlags);
	int (*xDelete)(system_vfs*, const char *zName, int syncDir);
	int (*xAccess)(system_vfs*, const char *zName, int flags, int *pResOut);
	int (*xFullPathname)(system_vfs*, const char *zName, int nOut, char *zOut);
	void *(*xDlOpen)(system_vfs*, const char *zFilename);
	void (*xDlError)(system_vfs*, int nByte, char *zErrMsg);
	void (*(*xDlSym)(system_vfs*,void*, const char *zSymbol))(void);
	void (*xDlClose)(system_vfs*, void*);
	int (*xRandomness)(system_vfs*, int nByte, char *zOut);
	int (*xSleep)(system_vfs*, int microseconds);
	int (*xCurrentTime)(system_vfs*, double*);
	int (*xGetLastError)(system_vfs*, int, char *);
	// The methods above are in version 1 of the sqlite_vfs object definition.  Those that follow are added in version 2 or later
	int (*xCurrentTimeInt64)(system_vfs*, INT64_TYPE*);
	// The methods above are in versions 1 and 2 of the sqlite_vfs object. New fields may be appended in figure versions.  The iVersion value will increment whenever this happens. 
};

/*
** API: Flags for the xAccess VFS method
**
** These integer constants can be used as the third parameter to the xAccess method of an [system_vfs] object.  They determine
** what kind of permissions the xAccess method is looking for. With SYSTEM_ACCESS_EXISTS, the xAccess method simply checks whether the
** file exists. With SYSTEM_ACCESS_READWRITE, the xAccess method checks whether the named directory is both readable and writable
** (in other words, if files can be added, removed, and renamed within the directory).
** The SYSTEM_ACCESS_READWRITE constant is currently used only by the [temp_store_directory pragma], though this could change in a future
** release of APPID. With SYSTEM_ACCESS_READ, the xAccess method checks whether the file is readable.  The SYSTEM_ACCESS_READ constant is
** currently unused, though it might be used in a future release of APPID.
*/
#define SYSTEM_ACCESS_EXISTS    0
#define SYSTEM_ACCESS_READWRITE 1   /* Used by PRAGMA temp_store_directory */
#define SYSTEM_ACCESS_READ      2   /* Unused */

/*
** API: Flags for the xShmLock VFS method
**
** These integer constants define the various locking operations allowed by the xShmLock method of [system_io_methods].  The
** following are the only legal combinations of flags to the xShmLock method:
**
** <ul>
** <li>  SYSTEM_SHM_LOCK | SYSTEM_SHM_SHARED
** <li>  SYSTEM_SHM_LOCK | SYSTEM_SHM_EXCLUSIVE
** <li>  SYSTEM_SHM_UNLOCK | SYSTEM_SHM_SHARED
** <li>  SYSTEM_SHM_UNLOCK | SYSTEM_SHM_EXCLUSIVE
** </ul>
**
** When unlocking, the same SHARED or EXCLUSIVE flag must be supplied as was given no the corresponding lock.  
**
** The xShmLock method can transition between unlocked and SHARED or between unlocked and EXCLUSIVE.  It cannot transition between SHARED
** and EXCLUSIVE.
*/
#define SYSTEM_SHM_UNLOCK       1
#define SYSTEM_SHM_LOCK         2
#define SYSTEM_SHM_SHARED       4
#define SYSTEM_SHM_EXCLUSIVE    8

/*
** API: Maximum xShmLock index
**
** The xShmLock method on [system_io_methods] may use values between 0 and this upper bound as its "offset" argument.
** The APPID core will never attempt to acquire or release a lock outside of this range
*/
#define SYSTEM_SHM_NLOCK        8

/*
** API: Virtual File System Objects
**
** A virtual filesystem (VFS) is an [system_vfs] object that APPID uses to interact with the underlying operating system.
** Most APPID builds come with a single default VFS that is appropriate for the host computer.
** New VFSes can be registered and existing VFSes can be unregistered. The following interfaces are provided.
**
** ^The system_vfs_find() interface returns a pointer to a VFS given its name. ^Names are case sensitive.
** ^Names are zero-terminated UTF-8 strings. ^If there is no match, a NULL pointer is returned. ^If zVfsName is NULL then the default VFS is returned.
**
** ^New VFSes are registered with system_vfs_register(). ^Each new VFS becomes the default VFS if the makeDflt flag is set.
** ^The same VFS can be registered multiple times without injury. ^To make an existing VFS into the default VFS, register it again
** with the makeDflt flag set.  If two different VFSes with the same name are registered, the behavior is undefined.  If a
** VFS is registered with a name that is NULL or an empty string, then the behavior is undefined.
**
** ^Unregister a VFS with the system_vfs_unregister() interface. ^(If the default VFS is unregistered, another VFS is chosen as
** the default.  The choice for the new VFS is arbitrary.)^
*/
SYSTEM_API system_vfs *system_vfs_find(const char *zVfsName);
SYSTEM_API int system_vfs_register(system_vfs*, int makeDflt);
SYSTEM_API int system_vfs_unregister(system_vfs*);


/* 
** APPCONTEXT ================================================================================================================================================
*/

/*
** API: Low-Level Control Of Database Files
**
** ^The [system_file_control()] interface makes a direct call to the xFileControl method for the [system_io_methods] object associated
** with a particular database identified by the second argument. ^The name of the database "main" for the main database or "temp" for the
** TEMP database, or the name that appears after the AS keyword for databases that are added using the [ATTACH] SQL command.
** ^A NULL pointer can be used in place of "main" to refer to the main database file. ^The third and fourth parameters to this routine
** are passed directly through to the second and third parameters of the xFileControl method.  ^The return value of the xFileControl
** method becomes the return value of this routine.
**
** ^If the second parameter (zDbName) does not match the name of any open database file, then SYSTEM_ERROR is returned.  ^This error
** code is not remembered and will not be recalled by [system_errcode()] or [system_errmsg()].  The underlying xFileControl method might
** also return SYSTEM_ERROR.  There is no way to distinguish between an incorrect zDbName and an SYSTEM_ERROR return from the underlying
** xFileControl method.
**
** See also: [SYSTEM_FCNTL_LOCKSTATE]
*/
SYSTEM_API int system_file_control(appContext*, const char *zDbName, int op, void*);

#ifdef __cplusplus
}  /* End of the 'extern "C"' block */
#endif
#endif  /* _SYSTEMAPI_IO_H_ */

