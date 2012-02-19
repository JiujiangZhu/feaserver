
// This is the header file for the generic hash-table implemenation used in APPID.
#ifndef _SYSTEM_HASH_H_
#define _SYSTEM_HASH_H_

// Forward declarations of structures.
typedef struct Hash Hash;
typedef struct HashElement HashElement;

// A complete hash table is an instance of the following structure. The internals of this structure are intended to be opaque -- client
// code should not attempt to access or modify the fields of this structure directly.  Change this structure only by using the routines below.
// However, some of the "procedures" and "functions" for modifying and accessing this structure are really macros, so we can't really make
// this structure opaque.
// 
// All elements of the hash table are on a single doubly-linked list. Hash.First points to the head of this list.
// 
// There are Hash.TableSize buckets.  Each bucket points to a spot in the global doubly-linked list.  The contents of the bucket are the
// element pointed to plus the next _hashTable.Count-1 elements in the list.
// 
// Hash.TableSize and Hash.Table may be zero.  In that case lookup is done by a linear search of the global list.  For small tables, the 
// Hash.Table table is never allocated because if there are few elements in the table, it is faster to do a linear search than to manage
// the hash table.
struct Hash {
	unsigned int TableSize;	// Number of buckets in the hash table
	unsigned int Count;		// Number of entries in this table
	HashElement *First;		// The first element of the array
	struct _hashTable {		// the hash table
		int Count;			// Number of entries with this hash
		HashElement *Chain;	// Pointer to first entry with this hash
	} *Table;
};

// Each element in the hash table is an instance of the following structure.  All elements are stored on a single doubly-linked list.
// Again, this structure is intended to be opaque, but it can't really be opaque because it is used by macros.
struct HashElement {
	HashElement *Next, *Prev;	// Next and previous elements in the table
	void *Data;					// Data associated with this element
	const char *pKey; int nKey;	// Key associated with this element
};

// Access routines.  To delete, insert a NULL pointer.
void systemHashInit(Hash*);
void *systemHashInsert(Hash*, const char *pKey, int nKey, void *pData);
void *systemHashFind(const Hash*, const char *pKey, int nKey);
void systemHashClear(Hash*);

/*
** Macros for looping over all elements of a hash table.  The idiom is
** like this:
**
**   Hash h;
**   HashElem *p;
**   ...
**   for(p=sqliteHashFirst(&h); p; p=sqliteHashNext(p)){
**     SomeStructure *pData = sqliteHashData(p);
**     // do something with pData
**   }
*/
#define systemHashFirst(H)  ((H)->First)
#define systemHashNext(E)   ((E)->Next)
#define systemHashData(E)   ((E)->Data)
/* #define systemHashKey(E)    ((E)->pKey) // NOT USED */
/* #define systemHashKeysize(E) ((E)->nKey)  // NOT USED */

// Number of entries in a hash table
/* #define systemHashCount(H)  ((H)->count) // NOT USED */

#endif /* _SYSTEM_HASH_H_ */
