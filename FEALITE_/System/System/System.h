#ifndef _SYSTEM_H_
#define _SYSTEM_H_
#include "System+Reset.h"
#include "System+Coverage.h"
#include "SystemApi+AppContext.h"
#include "System+Primitives.h"
#include "System+Memory.h"
#include "System+Threading.h"
#include "System+Config.h"
#include "System+Limits.h"
//
#include "SystemApi.h"
//
#include "System.OS.h"
#include "System.Hash.h"
#include "System+AppContext.h"
#include "../Parse/Parse.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>


int systemStatusValue(int);
void systemStatusAdd(int, int);
void systemStatusSet(int, int);

#endif /* _SYSTEM_H_ */
