// This is the implementation of generic hash-tables used in APPID.
#include "System.h"
#include <assert.h>

/* Turn bulk memory into a hash table object by initializing the
** fields of the Hash structure.
**
** "pNew" is a pointer to the hash table that is to be initialized.
*/
void systemHashInit(Hash *hash) {
	assert(hash != 0);
	hash->First = 0;
	hash->Count = 0;
	hash->TableSize = 0;
	hash->Table = 0;
}

// Remove all entries from a hash table.  Reclaim all memory.
// Call this routine to delete a hash table or to reset a hash table to the empty state.
void systemHashClear(Hash *hash) {
	HashElement *element;	// For looping over all elements of the table
	assert(hash != 0);
	element = hash->First;
	hash->First = 0;
	system_free(hash->Table); hash->Table = 0;
	hash->TableSize = 0;
	while(element) {
		HashElement *nextElement = element->Next;
		system_free(element);
		element = nextElement;
	}
	hash->Count = 0;
}

// The hashing function.
static unsigned int strHash(const char *z, int nKey){
	int h = 0;
	assert(nKey >= 0);
	while(nKey > 0) {
		h = ((h<<3)^h^systemUpperToLower[(unsigned char)*z++]);
		nKey--;
	}
	return h;
}

// Link pNew element into the hash table pH.  If pEntry!=0 then also insert pNew into the pEntry hash bucket.
static void insertElement (
	Hash *pH,			// The complete hash table
	struct _hashTable *pEntry,	// The entry into which pNew is inserted
	HashElement *pNew	// The element to be inserted
) {
	HashElement *pHead;       // First element already in pEntry
	if (pEntry){
		pHead = (pEntry->Count ? pEntry->Chain : 0);
		pEntry->Count++;
		pEntry->Chain = pNew;
	} else
		pHead = 0;
	if (pHead) {
		pNew->Next = pHead;
		pNew->Prev = pHead->Prev;
		if (pHead->Prev)
			pHead->Prev->Next = pNew;
		else
			pH->First = pNew;
		pHead->Prev = pNew;
	} else {
		pNew->Next = pH->First;
		if (pH->First)
			pH->First->Prev = pNew;
		pNew->Prev = 0;
		pH->First = pNew;
	}
}

// Resize the hash table so that it cantains "new_size" buckets.
// The hash table might fail to resize if system_malloc() fails or if the new size is the same as the prior size.
// Return TRUE if the resize occurs and false if not.
static int rehash(Hash *pH, unsigned int newSize) {
	struct _hashTable *newHashTable;				// The new hash table
	HashElement *element, *nextElement;	// For looping over existing elements
#if SYSTEM_MALLOC_SOFTLIMIT > 0
	if (newSize*sizeof(struct _hashTable) > SYSTEM_MALLOC_SOFTLIMIT)
		newSize = SYSTEM_MALLOC_SOFTLIMIT/sizeof(struct _hashTable);
	if (newSize == pH->TableSize)
		return 0;
#endif
	// The inability to allocates space for a larger hash table is a performance hit but it is not a fatal error.  So mark the allocation as a benign.
	systemBeginBenignMalloc();
	newHashTable = (struct _hashTable *)systemMalloc(newSize*sizeof(struct _hashTable));
	systemEndBenignMalloc();
	if (newHashTable == 0)
		return 0;
	system_free(pH->Table);
	pH->Table = newHashTable;
	pH->TableSize = newSize = systemMallocSize(newHashTable)/sizeof(struct _hashTable);
	memset(newHashTable, 0, newSize*sizeof(struct _hashTable));
	for(element = pH->First, pH->First = 0; element; element = nextElement) {
		unsigned int h = strHash(element->pKey, element->nKey) % newSize;
		nextElement = element->Next;
		insertElement(pH, &newHashTable[h], element);
	}
	return 1;
}

// This function (for internal use only) locates an element in an hash table that matches the given key.  The hash for this key has
// already been computed and is passed as the 4th parameter.
static HashElement *findElementGivenHash (
	const Hash *pH,     // The pH to be searched
	const char *pKey,   // The key we are searching for
	int nKey,           // Bytes in key (not counting zero terminator)
	unsigned int h      // The hash for this key.
){
	HashElement *element; // Used to loop thru the element list
	int count; // Number of elements left to test
	if (pH->Table) {
		struct _hashTable *pEntry = &pH->Table[h];
		element = pEntry->Chain;
		count = pEntry->Count;
	} else {
		element = pH->First;
		count = pH->Count;
	}
	while(count-- && ALWAYS(element)) {
		if ((element->nKey == nKey) && (systemStrNICmp(element->pKey, pKey, nKey) == 0))
			return element;
		element = element->Next;
	}
	return 0;
}

// Remove a single entry from the hash table given a pointer to that element and a hash on the element's key.
static void removeElementGivenHash (
	Hash *pH,				// The pH containing "elem"
	HashElement* element,	// The element to be removed from the pH
	unsigned int h		// Hash value for the element
) {
	struct _hashTable *pEntry;
	if (element->Prev)
		element->Prev->Next = element->Next; 
	else
		pH->First = element->Next;
	if (element->Next)
		element->Next->Prev = element->Prev;
	if (pH->Table) {
		pEntry = &pH->Table[h];
		if (pEntry->Chain == element)
			pEntry->Chain = element->Next;
		pEntry->Count--;
		assert(pEntry->Count >= 0);
	}
	system_free(element);
	pH->Count--;
	if (pH->Count <= 0) {
		assert(pH->first == 0);
		assert(pH->count == 0);
		systemHashClear(pH);
	}
}

// Attempt to locate an element of the hash table pH with a key that matches pKey,nKey.  Return the data for this element if it is
// found, or NULL if there is no match.
void *systemHashFind(const Hash *pH, const char *pKey, int nKey) {
	HashElement *element; // The element that matches key
	unsigned int h; // A hash on key
	assert(pH != 0);
	assert(pKey != 0);
	assert(nKey >= 0);
	h = (pH->Table ? strHash(pKey, nKey) % pH->TableSize : 0);
	element = findElementGivenHash(pH, pKey, nKey, h);
	return (element ? element->Data : 0);
}

/* Insert an element into the hash table pH.  The key is pKey,nKey
** and the data is "data".
**
** If no element exists with a matching key, then a new
** element is created and NULL is returned.
**
** If another element already exists with the same key, then the
** new data replaces the old data and the old data is returned.
** The key is not copied in this instance.  If a malloc fails, then
** the new data is returned and the hash table is unchanged.
**
** If the "data" parameter to this function is NULL, then the
** element corresponding to "key" is removed from the hash table.
*/
void *systemHashInsert(Hash *pH, const char *pKey, int nKey, void *data) {
	unsigned int h; // the hash of the key modulo hash table size
	HashElement *element; // Used to loop thru the element list
	HashElement *newElement; // New element added to the pH
	assert(pH != 0);
	assert(pKey != 0);
	assert(nKey >= 0);
	h = (pH->Table ? strHash(pKey, nKey) % pH->TableSize : 0);
	element = findElementGivenHash(pH, pKey, nKey, h);
	if (element) {
		void *lastData = element->Data;
		if (data == 0)
			removeElementGivenHash(pH, element, h);
		else {
			element->Data = data;
			element->pKey = pKey;
			assert(nKey == elem->nKey);
		}
		return lastData;
	}
	if (data == 0)
		return 0;
	newElement = (HashElement*)systemMalloc(sizeof(HashElement));
	if (newElement == 0)
		return data;
	newElement->pKey = pKey;
	newElement->nKey = nKey;
	newElement->Data = data;
	pH->Count++;
	if ((pH->Count >= 10) && pH->Count > (2*pH->TableSize))
		if (rehash(pH, pH->Count*2)) {
			assert(pH->TableSize > 0);
			h = strHash(pKey, nKey) % pH->TableSize;
		}
	if (pH->Table)
		insertElement(pH, &pH->Table[h], newElement);
	else
		insertElement(pH, 0, newElement);
	return 0;
}
