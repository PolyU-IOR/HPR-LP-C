#include "mps_reader.h"
#include "utils.h"
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <iostream>
#include <iomanip>

#define INITIAL_CAPACITY 8192
#define INITIAL_NNZ_CAPACITY 100000

/* Fast HPRLP_FLOAT parsing - atof is actually quite optimized in modern libc,
   but we can add inline hint */
static inline HPRLP_FLOAT parse_double(const char *str) {
    return atof(str);
}

/* ============================================================================
 * Utility Functions
 * ========================================================================== */

/* Safe string duplication */
char* strdup_safe(const char *str) {
    if (!str) return NULL;
    size_t len = strlen(str);
    char *dup = (char*)malloc(len + 1);
    if (dup) {
        memcpy(dup, str, len + 1);  /* memcpy is faster than strcpy */
    }
    return dup;
}

/* Trim leading and trailing whitespace in-place, return pointer to start */
char* trim_inplace(char *str) {
    if (!str) return NULL;
    
    /* Trim leading whitespace */
    while (*str && isspace((unsigned char)*str)) str++;
    
    if (*str == '\0') return str;  /* All spaces */
    
    /* Trim trailing whitespace */
    char *end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    
    /* Write new null terminator */
    *(end + 1) = '\0';
    
    return str;
}

/* Original trim function for backward compatibility */
void trim(char *str) {
    if (!str) return;
    
    char *trimmed = trim_inplace(str);
    if (trimmed != str) {
        memmove(str, trimmed, strlen(trimmed) + 1);
    }
}

/* ============================================================================
 * Name-Index Map Functions (with Hash Table)
 * ========================================================================== */

/* Simple hash function for strings */
static unsigned int hash_string(const char *str, int num_buckets) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash % num_buckets;
}

void namemap_init(NameIndexMap *map) {
    map->size = 0;
    map->capacity = 1024;  /* Start with larger capacity */
    map->num_buckets = 4096;  /* Power of 2 for better performance */
    map->pairs = (NameIndexPair*)malloc(map->capacity * sizeof(NameIndexPair));
    map->buckets = (int*)malloc(map->num_buckets * sizeof(int));
    
    /* Initialize all buckets to -1 (empty) */
    for (int i = 0; i < map->num_buckets; i++) {
        map->buckets[i] = -1;
    }
}

void namemap_free(NameIndexMap *map) {
    for (int i = 0; i < map->size; i++) {
        free(map->pairs[i].name);
    }
    free(map->pairs);
    free(map->buckets);
    map->pairs = NULL;
    map->buckets = NULL;
    map->size = 0;
    map->capacity = 0;
    map->num_buckets = 0;
}

int namemap_get(NameIndexMap *map, const char *name, int default_value) {
    if (map->size == 0) return default_value;
    
    unsigned int bucket = hash_string(name, map->num_buckets);
    int idx = map->buckets[bucket];
    
    /* Follow the chain */
    while (idx >= 0) {
        if (strcmp(map->pairs[idx].name, name) == 0) {
            return map->pairs[idx].index;
        }
        idx = map->pairs[idx].next;
    }
    
    return default_value;
}

void namemap_set(NameIndexMap *map, const char *name, int index) {
    /* Check if already exists */
    unsigned int bucket = hash_string(name, map->num_buckets);
    int idx = map->buckets[bucket];
    
    while (idx >= 0) {
        if (strcmp(map->pairs[idx].name, name) == 0) {
            map->pairs[idx].index = index;
            return;
        }
        idx = map->pairs[idx].next;
    }
    
    /* Add new entry */
    if (map->size >= map->capacity) {
        map->capacity *= 2;
        map->pairs = (NameIndexPair*)realloc(map->pairs, map->capacity * sizeof(NameIndexPair));
        
        /* Rehash if load factor is too high */
        if (map->size > map->num_buckets * 0.75) {
            map->num_buckets *= 2;
            map->buckets = (int*)realloc(map->buckets, map->num_buckets * sizeof(int));
            
            /* Initialize new buckets */
            for (int i = 0; i < map->num_buckets; i++) {
                map->buckets[i] = -1;
            }
            
            /* Rehash all entries */
            for (int i = 0; i < map->size; i++) {
                unsigned int new_bucket = hash_string(map->pairs[i].name, map->num_buckets);
                map->pairs[i].next = map->buckets[new_bucket];
                map->buckets[new_bucket] = i;
            }
            
            /* Recalculate bucket for the new entry */
            bucket = hash_string(name, map->num_buckets);
        }
    }
    
    int new_idx = map->size;
    map->pairs[new_idx].name = strdup_safe(name);
    map->pairs[new_idx].index = index;
    map->pairs[new_idx].next = map->buckets[bucket];
    map->buckets[bucket] = new_idx;
    map->size++;
}

/* ============================================================================
 * QPSData Functions
 * ========================================================================== */

/* Estimate problem size from file size (heuristic) */
static void estimate_problem_size(FILE *fp, int *est_vars, int *est_cons, int *est_nnz) {
    long file_size = -1;
    
    /* Get file size using portable fseek/ftell */
    long current_pos = ftell(fp);
    if (current_pos >= 0) {
        if (fseek(fp, 0, SEEK_END) == 0) {
            file_size = ftell(fp);
            /* Restore original position */
            fseek(fp, current_pos, SEEK_SET);
        }
    }
    
    if (file_size > 0) {
        /* Rough heuristic: assume ~50 bytes per coefficient entry */
        *est_nnz = (int)(file_size / 50);
        /* Assume sparse matrices with ~10 entries per variable */
        *est_vars = (*est_nnz) / 10;
        *est_cons = (*est_vars) / 2;  /* Rough estimate */
        
        /* Clamp to reasonable ranges */
        if (*est_vars < INITIAL_CAPACITY) *est_vars = INITIAL_CAPACITY;
        if (*est_cons < INITIAL_CAPACITY) *est_cons = INITIAL_CAPACITY;
        if (*est_nnz < INITIAL_NNZ_CAPACITY) *est_nnz = INITIAL_NNZ_CAPACITY;
        
        /* Cap at reasonable maximums to avoid over-allocation */
        if (*est_vars > 1000000) *est_vars = 1000000;
        if (*est_cons > 1000000) *est_cons = 1000000;
        if (*est_nnz > 10000000) *est_nnz = 10000000;
    } else {
        *est_vars = INITIAL_CAPACITY;
        *est_cons = INITIAL_CAPACITY;
        *est_nnz = INITIAL_NNZ_CAPACITY;
    }
}

QPSData* qpsdata_create(void) {
    return qpsdata_create_with_capacity(INITIAL_CAPACITY, INITIAL_CAPACITY, INITIAL_NNZ_CAPACITY);
}

QPSData* qpsdata_create_with_capacity(int var_cap, int con_cap, int nnz_cap) {
    QPSData *qps = (QPSData*)calloc(1, sizeof(QPSData));
    if (!qps) return NULL;
    
    qps->nvar = 0;
    qps->ncon = 0;
    qps->objsense = OBJSENSE_NOTSET;
    qps->c0 = 0.0;
    
    /* Set capacities */
    qps->var_capacity = var_cap;
    qps->con_capacity = con_cap;
    qps->qnnz_capacity = nnz_cap;
    qps->annz_capacity = nnz_cap;
    
    /* Allocate arrays */
    qps->c = (HPRLP_FLOAT*)calloc(qps->var_capacity, sizeof(HPRLP_FLOAT));
    qps->lvar = (HPRLP_FLOAT*)malloc(qps->var_capacity * sizeof(HPRLP_FLOAT));
    qps->uvar = (HPRLP_FLOAT*)malloc(qps->var_capacity * sizeof(HPRLP_FLOAT));
    qps->varnames = (char**)calloc(qps->var_capacity, sizeof(char*));
    qps->vartypes = (VariableType*)calloc(qps->var_capacity, sizeof(VariableType));
    
    qps->lcon = (HPRLP_FLOAT*)malloc(qps->con_capacity * sizeof(HPRLP_FLOAT));
    qps->ucon = (HPRLP_FLOAT*)malloc(qps->con_capacity * sizeof(HPRLP_FLOAT));
    qps->connames = (char**)calloc(qps->con_capacity, sizeof(char*));
    qps->contypes = (RowType*)calloc(qps->con_capacity, sizeof(RowType));
    
    qps->qrows = (int*)malloc(qps->qnnz_capacity * sizeof(int));
    qps->qcols = (int*)malloc(qps->qnnz_capacity * sizeof(int));
    qps->qvals = (HPRLP_FLOAT*)malloc(qps->qnnz_capacity * sizeof(HPRLP_FLOAT));
    qps->qnnz = 0;
    
    qps->arows = (int*)malloc(qps->annz_capacity * sizeof(int));
    qps->acols = (int*)malloc(qps->annz_capacity * sizeof(int));
    qps->avals = (HPRLP_FLOAT*)malloc(qps->annz_capacity * sizeof(HPRLP_FLOAT));
    qps->annz = 0;
    
    /* Initialize name maps */
    namemap_init(&qps->varindices);
    namemap_init(&qps->conindices);
    
    /* Initialize NaN for bounds (will be set to defaults later) */
    for (int i = 0; i < qps->var_capacity; i++) {
        qps->lvar[i] = NAN;
        qps->uvar[i] = NAN;
    }
    
    return qps;
}

void qpsdata_free(QPSData *qps) {
    if (!qps) return;
    
    free(qps->c);
    free(qps->lvar);
    free(qps->uvar);
    free(qps->lcon);
    free(qps->ucon);
    
    free(qps->qrows);
    free(qps->qcols);
    free(qps->qvals);
    
    free(qps->arows);
    free(qps->acols);
    free(qps->avals);
    
    free(qps->vartypes);
    free(qps->contypes);
    
    /* Free names */
    free(qps->name);
    free(qps->objname);
    free(qps->rhsname);
    free(qps->bndname);
    free(qps->rngname);
    
    for (int i = 0; i < qps->nvar; i++) {
        free(qps->varnames[i]);
    }
    free(qps->varnames);
    
    for (int i = 0; i < qps->ncon; i++) {
        free(qps->connames[i]);
    }
    free(qps->connames);
    
    namemap_free(&qps->varindices);
    namemap_free(&qps->conindices);
    
    free(qps);
}

/* ============================================================================
 * Card Parsing Functions
 * ========================================================================== */

/* Parse a line in fixed MPS format */
void read_card_fixed(MPSCard *card, const char *line) {
    int len = strlen(line);
    
    /* Initialize card */
    card->iscomment = false;
    card->isheader = false;
    card->nfields = 0;
    memset(card->f1, 0, sizeof(card->f1));
    memset(card->f2, 0, sizeof(card->f2));
    memset(card->f3, 0, sizeof(card->f3));
    memset(card->f4, 0, sizeof(card->f4));
    memset(card->f5, 0, sizeof(card->f5));
    memset(card->f6, 0, sizeof(card->f6));
    
    /* Check for empty line or comment */
    if (len == 0 || line[0] == '*' || line[0] == '&') {
        card->iscomment = true;
        return;
    }
    
    /* Check for section header */
    if (!isspace((unsigned char)line[0])) {
        card->isheader = true;
        card->nfields = 1;
        
        /* Parse first field manually (portable alternative to sscanf) */
        int i = 0;
        while (i < (int)sizeof(card->f1) - 1 && line[i] && !isspace((unsigned char)line[i])) {
            card->f1[i] = line[i];
            i++;
        }
        card->f1[i] = '\0';
        
        /* Check for NAME section */
        if (strcmp(card->f1, "NAME") == 0 && len >= 15) {
            strncpy(card->f2, line + 14, sizeof(card->f2) - 1);
            trim(card->f2);
            card->nfields = 2;
        }
        
        /* Check for OBJECT BOUND */
        if (strcmp(card->f1, "OBJECT") == 0) {
            char temp[256];
            /* Parse second field manually */
            int j = i;
            while (j < (int)len && isspace((unsigned char)line[j])) j++;
            int k = 0;
            while (j < (int)len && k < 255 && !isspace((unsigned char)line[j])) {
                temp[k++] = line[j++];
            }
            temp[k] = '\0';
            if (k > 0 && strcmp(temp, "BOUND") == 0) {
                strcpy(card->f1, "OBJECT BOUND");
            }
        }
        return;
    }
    
    /* Regular card - Fixed format parsing */
    /* Field 1: columns 2-3 */
    if (len >= 3) {
        strncpy(card->f1, line + 1, 2);
        card->f1[2] = '\0';
        trim(card->f1);
        if (strlen(card->f1) > 0) card->nfields = 1;
    }
    
    /* Field 2: columns 5-12 */
    if (len >= 5) {
        int end = (len >= 12) ? 12 : len;
        strncpy(card->f2, line + 4, end - 4);
        card->f2[end - 4] = '\0';
        trim(card->f2);
        if (strlen(card->f2) > 0) card->nfields = 2;
    }
    
    /* Field 3: columns 15-22 */
    if (len >= 15) {
        int end = (len >= 22) ? 22 : len;
        strncpy(card->f3, line + 14, end - 14);
        card->f3[end - 14] = '\0';
        trim(card->f3);
        if (strlen(card->f3) > 0) card->nfields = 3;
    }
    
    /* Field 4: columns 25-36 */
    if (len >= 25) {
        int end = (len >= 36) ? 36 : len;
        strncpy(card->f4, line + 24, end - 24);
        card->f4[end - 24] = '\0';
        trim(card->f4);
        if (strlen(card->f4) > 0) card->nfields = 4;
    }
    
    /* Field 5: columns 40-47 */
    if (len >= 40) {
        int end = (len >= 47) ? 47 : len;
        strncpy(card->f5, line + 39, end - 39);
        card->f5[end - 39] = '\0';
        trim(card->f5);
        if (strlen(card->f5) > 0) card->nfields = 5;
    }
    
    /* Field 6: columns 50-61 */
    if (len >= 50) {
        int end = (len >= 61) ? 61 : len;
        strncpy(card->f6, line + 49, end - 49);
        card->f6[end - 49] = '\0';
        trim(card->f6);
        if (strlen(card->f6) > 0) card->nfields = 6;
    }
    
    /* If first field is empty, shift all fields left */
    if (card->nfields > 0 && strlen(card->f1) == 0) {
        strcpy(card->f1, card->f2);
        strcpy(card->f2, card->f3);
        strcpy(card->f3, card->f4);
        strcpy(card->f4, card->f5);
        strcpy(card->f5, card->f6);
        card->f6[0] = '\0';
        card->nfields--;
    }
}

/* Parse a line in free MPS format */
void read_card_free(MPSCard *card, const char *line) {
    /* Initialize card */
    card->iscomment = false;
    card->isheader = false;
    card->nfields = 0;
    memset(card->f1, 0, sizeof(card->f1));
    memset(card->f2, 0, sizeof(card->f2));
    memset(card->f3, 0, sizeof(card->f3));
    memset(card->f4, 0, sizeof(card->f4));
    memset(card->f5, 0, sizeof(card->f5));
    memset(card->f6, 0, sizeof(card->f6));
    
    int len = strlen(line);
    
    /* Check for empty line or comment */
    if (len == 0 || line[0] == '*' || line[0] == '&') {
        card->iscomment = true;
        return;
    }
    
    /* Check for section header */
    if (!isspace((unsigned char)line[0])) {
        card->isheader = true;
        
        char linecopy[1024];
        strncpy(linecopy, line, sizeof(linecopy) - 1);
        linecopy[sizeof(linecopy) - 1] = '\0';
        
        char *token = strtok(linecopy, " \t");
        if (token) {
            strcpy(card->f1, token);
            card->nfields = 1;
            
            /* Check for NAME section */
            if (strcmp(card->f1, "NAME") == 0) {
                token = strtok(NULL, " \t");
                if (token) {
                    strcpy(card->f2, token);
                    card->nfields = 2;
                }
            }
            
            /* Check for OBJECT BOUND */
            if (strcmp(card->f1, "OBJECT") == 0) {
                token = strtok(NULL, " \t");
                if (token && strcmp(token, "BOUND") == 0) {
                    strcpy(card->f1, "OBJECT BOUND");
                }
            }
        }
        return;
    }
    
    /* Regular card - Free format parsing */
    char linecopy[1024];
    strncpy(linecopy, line, sizeof(linecopy) - 1);
    linecopy[sizeof(linecopy) - 1] = '\0';
    
    char *fields[6] = {card->f1, card->f2, card->f3, card->f4, card->f5, card->f6};
    char *token = strtok(linecopy, " \t");
    int i = 0;
    
    while (token && i < 6) {
        strcpy(fields[i], token);
        i++;
        token = strtok(NULL, " \t");
    }
    
    card->nfields = i;
}

/* Get section type from header name */
SectionType get_section_type(const char *header) {
    if (strcmp(header, "ENDATA") == 0) return SECTION_ENDATA;
    if (strcmp(header, "NAME") == 0) return SECTION_NAME;
    if (strcmp(header, "OBJSENSE") == 0) return SECTION_OBJSENSE;
    if (strcmp(header, "ROWS") == 0) return SECTION_ROWS;
    if (strcmp(header, "COLUMNS") == 0) return SECTION_COLUMNS;
    if (strcmp(header, "RHS") == 0) return SECTION_RHS;
    if (strcmp(header, "BOUNDS") == 0) return SECTION_BOUNDS;
    if (strcmp(header, "RANGES") == 0) return SECTION_RANGES;
    if (strcmp(header, "QUADOBJ") == 0) return SECTION_QUADOBJ;
    if (strcmp(header, "QMATRIX") == 0) return SECTION_QMATRIX;
    if (strcmp(header, "OBJECT BOUND") == 0) return SECTION_OBJECT_BOUND;
    return SECTION_NONE;
}

/* ============================================================================
 * Section Reading Functions
 * ========================================================================== */

/* Read objective sense line */
void read_objsense_line(QPSData *qps, MPSCard *card) {
    if (strcmp(card->f1, "MIN") == 0) {
        qps->objsense = OBJSENSE_MIN;
    } else if (strcmp(card->f1, "MAX") == 0) {
        qps->objsense = OBJSENSE_MAX;
    } else {
        std::cerr << "Warning: Unrecognized objective sense: " << card->f1 << "\n";
    }
}

/* Get row type from character */
RowType get_row_type(const char *type_str) {
    if (strcmp(type_str, "N") == 0) return RTYPE_OBJECTIVE;
    if (strcmp(type_str, "E") == 0) return RTYPE_EQUALTO;
    if (strcmp(type_str, "L") == 0) return RTYPE_LESSTHAN;
    if (strcmp(type_str, "G") == 0) return RTYPE_GREATERTHAN;
    return RTYPE_OBJECTIVE; /* Default */
}

/* Read ROWS section line */
void read_rows_line(QPSData *qps, MPSCard *card) {
    if (card->nfields < 2) {
        std::cerr << "Error: Line " << card->nline << " contains only " << card->nfields << " fields\n";
        return;
    }
    
    RowType rtype = get_row_type(card->f1);
    char *rowname = card->f2;
    
    if (rtype == RTYPE_OBJECTIVE) {
        /* Objective row */
        if (qps->objname == NULL) {
            qps->objname = strdup_safe(rowname);
            namemap_set(&qps->conindices, rowname, 0);
            // std::cout << "Info: Using '" << rowname << "' as objective (l. " << card->nline << ")\n";
        } else {
            /* Rim objective */
            std::cerr << "Warning: Detected rim objective row " << rowname << " at line " << card->nline << "\n";
            namemap_set(&qps->conindices, rowname, -1);
        }
        return;
    }
    
    /* Regular constraint row */
    int ncon = qps->ncon;
    int con_index = ncon + 1;  /* Constraints start at index 1 (0 is for objective) */
    
    /* Check for duplicate */
    int existing = namemap_get(&qps->conindices, rowname, -2);
    if (existing == con_index) {
        std::cerr << "Error: Duplicate row name " << rowname << " at line " << card->nline << "\n";
        return;
    }
    
    /* Expand arrays if needed */
    if (ncon >= qps->con_capacity) {
        qps->con_capacity *= 2;
        qps->lcon = (HPRLP_FLOAT*)realloc(qps->lcon, qps->con_capacity * sizeof(HPRLP_FLOAT));
        qps->ucon = (HPRLP_FLOAT*)realloc(qps->ucon, qps->con_capacity * sizeof(HPRLP_FLOAT));
        qps->connames = (char**)realloc(qps->connames, qps->con_capacity * sizeof(char*));
        qps->contypes = (RowType*)realloc(qps->contypes, qps->con_capacity * sizeof(RowType));
    }
    
    namemap_set(&qps->conindices, rowname, con_index);
    qps->connames[ncon] = strdup_safe(rowname);
    qps->contypes[ncon] = rtype;
    
    /* Set default bounds based on row type */
    if (rtype == RTYPE_EQUALTO) {
        qps->lcon[ncon] = 0.0;
        qps->ucon[ncon] = 0.0;
    } else if (rtype == RTYPE_GREATERTHAN) {
        qps->lcon[ncon] = 0.0;
        qps->ucon[ncon] = INFINITY;
    } else if (rtype == RTYPE_LESSTHAN) {
        qps->lcon[ncon] = -INFINITY;
        qps->ucon[ncon] = 0.0;
    }
    
    qps->ncon++;
}

/* Add entry to constraint matrix A */
void add_constraint_entry(QPSData *qps, int row, int col, HPRLP_FLOAT val) {
    if (qps->annz >= qps->annz_capacity) {
        qps->annz_capacity *= 2;
        qps->arows = (int*)realloc(qps->arows, qps->annz_capacity * sizeof(int));
        qps->acols = (int*)realloc(qps->acols, qps->annz_capacity * sizeof(int));
        qps->avals = (HPRLP_FLOAT*)realloc(qps->avals, qps->annz_capacity * sizeof(HPRLP_FLOAT));
    }
    
    qps->arows[qps->annz] = row;
    qps->acols[qps->annz] = col;
    qps->avals[qps->annz] = val;
    qps->annz++;
}

/* Read COLUMNS section line - optimized version */
void read_columns_line(QPSData *qps, MPSCard *card, bool integer_section) {
    if (card->nfields < 3) {
        std::cerr << "Error: Line " << card->nline << " contains only " << card->nfields << " fields\n";
        return;
    }
    
    char *varname = card->f1;
    int nvar = qps->nvar;
    
    /* Get or create variable */
    int col = namemap_get(&qps->varindices, varname, nvar);
    if (col == nvar) {
        /* New variable */
        if (nvar >= qps->var_capacity) {
            qps->var_capacity *= 2;
            qps->c = (HPRLP_FLOAT*)realloc(qps->c, qps->var_capacity * sizeof(HPRLP_FLOAT));
            qps->lvar = (HPRLP_FLOAT*)realloc(qps->lvar, qps->var_capacity * sizeof(HPRLP_FLOAT));
            qps->uvar = (HPRLP_FLOAT*)realloc(qps->uvar, qps->var_capacity * sizeof(HPRLP_FLOAT));
            qps->varnames = (char**)realloc(qps->varnames, qps->var_capacity * sizeof(char*));
            qps->vartypes = (VariableType*)realloc(qps->vartypes, qps->var_capacity * sizeof(VariableType));
            
            /* Initialize new entries */
            for (int i = nvar; i < qps->var_capacity; i++) {
                qps->lvar[i] = NAN;
                qps->uvar[i] = NAN;
                qps->c[i] = 0.0;
            }
        }
        
        namemap_set(&qps->varindices, varname, nvar);
        qps->varnames[nvar] = strdup_safe(varname);
        qps->c[nvar] = 0.0;
        qps->vartypes[nvar] = integer_section ? VTYPE_MARKED : VTYPE_CONTINUOUS;
        qps->lvar[nvar] = NAN;
        qps->uvar[nvar] = NAN;
        qps->nvar++;
    }
    
    /* First coefficient pair */
    char *rowname1 = card->f2;
    HPRLP_FLOAT val1 = atof(card->f3);
    
    int row1 = namemap_get(&qps->conindices, rowname1, -2);
    if (row1 == 0) {
        /* Objective */
        qps->c[col] = val1;
    } else if (row1 > 0) {
        /* Regular constraint */
        add_constraint_entry(qps, row1 - 1, col, val1);
    } else if (row1 != -1) {
        std::cerr << "Error: Unknown row " << rowname1 << " at line " << card->nline << "\n";
    }
    
    /* Second coefficient pair (optional) */
    if (card->nfields >= 5) {
        char *rowname2 = card->f4;
        HPRLP_FLOAT val2 = atof(card->f5);
        
        int row2 = namemap_get(&qps->conindices, rowname2, -2);
        if (row2 == 0) {
            qps->c[col] = val2;
        } else if (row2 > 0) {
            add_constraint_entry(qps, row2 - 1, col, val2);
        } else if (row2 != -1) {
            std::cerr << "Error: Unknown row " << rowname2 << " at line " << card->nline << "\n";
        }
    }
}

/* Read RHS section line */
void read_rhs_line(QPSData *qps, MPSCard *card) {
    if (card->nfields < 3) {
        std::cerr << "Error: Line " << card->nline << " contains only " << card->nfields << " fields\n";
        return;
    }
    
    char *rhs = card->f1;
    if (qps->rhsname == NULL) {
        qps->rhsname = strdup_safe(rhs);
        // std::cout << "Info: Using '" << rhs << "' as RHS (l. " << card->nline << ")\n";
    } else if (strcmp(qps->rhsname, rhs) != 0) {
        std::cerr << "Error: Skipping line " << card->nline << " with rim RHS " << rhs << "\n";
        return;
    }
    
    /* First RHS value */
    char *rowname = card->f2;
    HPRLP_FLOAT val = atof(card->f3);
    
    int row = namemap_get(&qps->conindices, rowname, -2);
    if (row == 0) {
        /* Objective constant */
        qps->c0 = -val;
    } else if (row == -1) {
        std::cerr << "Error: Ignoring RHS for rim objective " << rowname << " at line " << card->nline << "\n";
    } else if (row > 0) {
        int idx = row - 1;
        RowType rtype = qps->contypes[idx];
        if (rtype == RTYPE_EQUALTO) {
            qps->lcon[idx] = val;
            qps->ucon[idx] = val;
        } else if (rtype == RTYPE_LESSTHAN) {
            qps->ucon[idx] = val;
        } else if (rtype == RTYPE_GREATERTHAN) {
            qps->lcon[idx] = val;
        }
    } else {
        std::cerr << "Error: Unknown row " << rowname << "\n";
    }
    
    /* Second RHS value (optional) */
    if (card->nfields >= 5) {
        rowname = card->f4;
        val = atof(card->f5);
        
        row = namemap_get(&qps->conindices, rowname, -2);
        if (row == 0) {
            qps->c0 = -val;
        } else if (row == -1) {
            std::cerr << "Error: Ignoring RHS for rim objective " << rowname << " at line " << card->nline << "\n";
        } else if (row > 0) {
            int idx = row - 1;
            RowType rtype = qps->contypes[idx];
            if (rtype == RTYPE_EQUALTO) {
                qps->lcon[idx] = val;
                qps->ucon[idx] = val;
            } else if (rtype == RTYPE_LESSTHAN) {
                qps->ucon[idx] = val;
            } else if (rtype == RTYPE_GREATERTHAN) {
                qps->lcon[idx] = val;
            }
        } else {
            std::cerr << "Error: Unknown row " << rowname << "\n";
        }
    }
}

/* Helper function to apply a range value to a constraint */
static void apply_range_to_constraint(QPSData *qps, const char *rowname, HPRLP_FLOAT val, int line_num) {
    int row = namemap_get(&qps->conindices, rowname, -2);
    if (row == 0 || row == -1) {
        std::cerr << "Error: Encountered objective row " << rowname << " in RANGES section (l. " << line_num << ")\n";
        return;
    } else if (row > 0) {
        int idx = row - 1;
        RowType rtype = qps->contypes[idx];
        
        if (rtype == RTYPE_EQUALTO) {
            if (val >= 0.0) {
                qps->ucon[idx] += val;
            } else {
                qps->lcon[idx] += val;
            }
        } else if (rtype == RTYPE_LESSTHAN) {
            qps->lcon[idx] = qps->ucon[idx] - fabs(val);
        } else if (rtype == RTYPE_GREATERTHAN) {
            qps->ucon[idx] = qps->lcon[idx] + fabs(val);
        }
    } else {
        std::cerr << "Error: Unknown row " << rowname << " in RANGES section (l. " << line_num << ")\n";
    }
}

/* Read RANGES section line */
void read_ranges_line(QPSData *qps, MPSCard *card) {
    if (card->nfields < 3) {
        std::cerr << "Error: Line " << card->nline << " contains only " << card->nfields << " fields\n";
        return;
    }
    
    char *rng = card->f1;
    if (qps->rngname == NULL) {
        qps->rngname = strdup_safe(rng);
        // std::cout << "Info: Using '" << rng << "' as RANGES (l. " << card->nline << ")\n";
    } else if (strcmp(qps->rngname, rng) != 0) {
        std::cerr << "Error: Skipping line " << card->nline << " with rim RANGES " << rng << "\n";
        return;
    }
    
    /* Process first constraint-value pair */
    char *rowname = card->f2;
    HPRLP_FLOAT val = atof(card->f3);
    apply_range_to_constraint(qps, rowname, val, card->nline);
    
    /* Process second constraint-value pair if present */
    if (card->nfields >= 5 && strlen(card->f4) > 0) {
        char *rowname2 = card->f4;
        HPRLP_FLOAT val2 = atof(card->f5);
        apply_range_to_constraint(qps, rowname2, val2, card->nline);
    }
}

/* Read BOUNDS section line */
void read_bounds_line(QPSData *qps, MPSCard *card) {
    if (card->nfields < 3) {
        std::cerr << "Error: Line " << card->nline << " contains only " << card->nfields << " fields\n";
        return;
    }
    
    char *bnd = card->f2;
    if (qps->bndname == NULL) {
        qps->bndname = strdup_safe(bnd);
        // std::cout << "Info: Using '" << bnd << "' as BOUNDS (l. " << card->nline << ")\n";
    } else if (strcmp(qps->bndname, bnd) != 0) {
        std::cerr << "Error: Skipping line " << card->nline << " with rim bound " << bnd << "\n";
        return;
    }
    
    char *varname = card->f3;
    int col = namemap_get(&qps->varindices, varname, -1);
    if (col < 0) {
        std::cerr << "Error: Unknown column " << varname << "\n";
        return;
    }
    
    char *btype = card->f1;
    
    /* Bounds that don't require a value */
    if (strcmp(btype, "FR") == 0) {
        qps->lvar[col] = -INFINITY;
        qps->uvar[col] = INFINITY;
        return;
    } else if (strcmp(btype, "MI") == 0) {
        qps->lvar[col] = -INFINITY;
        return;
    } else if (strcmp(btype, "PL") == 0) {
        qps->uvar[col] = INFINITY;
        return;
    } else if (strcmp(btype, "BV") == 0) {
        qps->vartypes[col] = VTYPE_BINARY;
        qps->lvar[col] = 0.0;
        qps->uvar[col] = 1.0;
        return;
    }
    
    /* Bounds that require a value */
    if (card->nfields < 4) {
        std::cerr << "Error: At least 4 fields required for " << btype << " bounds\n";
        return;
    }
    
    HPRLP_FLOAT val = atof(card->f4);
    
    if (strcmp(btype, "LO") == 0) {
        qps->lvar[col] = val;
    } else if (strcmp(btype, "UP") == 0) {
        qps->uvar[col] = val;
    } else if (strcmp(btype, "FX") == 0) {
        qps->lvar[col] = val;
        qps->uvar[col] = val;
    } else if (strcmp(btype, "LI") == 0) {
        qps->vartypes[col] = VTYPE_INTEGER;
        qps->lvar[col] = val;
    } else if (strcmp(btype, "UI") == 0) {
        qps->vartypes[col] = VTYPE_INTEGER;
        qps->uvar[col] = val;
    } else {
        std::cerr << "Warning: Unknown bound type " << btype << "\n";
    }
}

/* Read QUADOBJ section line */
void read_quadobj_line(QPSData *qps, MPSCard *card) {
    if (card->nfields < 3) {
        std::cerr << "Error: Line " << card->nline << " contains only " << card->nfields << " fields\n";
        return;
    }
    
    char *colname1 = card->f1;
    char *colname2 = card->f2;
    HPRLP_FLOAT val = atof(card->f3);
    
    int col1 = namemap_get(&qps->varindices, colname1, -1);
    int col2 = namemap_get(&qps->varindices, colname2, -1);
    
    if (col1 < 0) {
        std::cerr << "Error: Unknown variable " << colname1 << "\n";
        return;
    }
    if (col2 < 0) {
        std::cerr << "Error: Unknown variable " << colname2 << "\n";
        return;
    }
    
    /* Expand arrays if needed */
    if (qps->qnnz >= qps->qnnz_capacity) {
        qps->qnnz_capacity *= 2;
        qps->qrows = (int*)realloc(qps->qrows, qps->qnnz_capacity * sizeof(int));
        qps->qcols = (int*)realloc(qps->qcols, qps->qnnz_capacity * sizeof(int));
        qps->qvals = (HPRLP_FLOAT*)realloc(qps->qvals, qps->qnnz_capacity * sizeof(HPRLP_FLOAT));
    }
    
    qps->qrows[qps->qnnz] = col1;
    qps->qcols[qps->qnnz] = col2;
    qps->qvals[qps->qnnz] = val;
    qps->qnnz++;
}

/* ============================================================================
 * Main Reading Function
 * ========================================================================== */

QPSData* readqps_from_file(FILE *fp, MPSFormat format) {
    /* Estimate problem size from file */
    int est_vars, est_cons, est_nnz;
    estimate_problem_size(fp, &est_vars, &est_cons, &est_nnz);
    
    QPSData *qps = qpsdata_create_with_capacity(est_vars, est_cons, est_nnz);
    if (!qps) return NULL;
    
    /* Use larger buffer for better I/O performance */
    char *buffer = (char*)malloc(65536);  /* 64KB buffer */
    if (buffer) {
        setvbuf(fp, buffer, _IOFBF, 65536);
    }
    
    MPSCard card;
    card.nline = 0;
    
    bool name_section_read = false;
    bool objsense_section_read = false;
    bool rows_section_read = false;
    bool columns_section_read = false;
    bool rhs_section_read = false;
    bool bounds_section_read = false;
    bool ranges_section_read = false;
    bool quadobj_section_read = false;
    bool endata_read = false;
    
    bool integer_section = false;
    
    SectionType current_section = SECTION_NONE;
    
    char line[1024];
    
    while (fgets(line, sizeof(line), fp)) {
        card.nline++;
        
        /* Remove newline */
        line[strcspn(line, "\r\n")] = 0;
        
        /* Parse card based on format */
        if (format == MPS_FIXED) {
            read_card_fixed(&card, line);
        } else {
            read_card_free(&card, line);
        }
        
        /* Skip comments and empty lines */
        if (card.iscomment) continue;
        
        /* Handle section headers */
        if (card.isheader) {
            SectionType sec = get_section_type(card.f1);
            
            if (sec == SECTION_NAME) {
                if (name_section_read) {
                    std::cerr << "Error: More than one NAME section\n";
                    goto error;
                }
                qps->name = strdup_safe(card.f2);
                name_section_read = true;
                // std::cout << "Info: Using '" << qps->name << "' as NAME (l. " << card.nline << ")\n";
            } else if (sec == SECTION_OBJSENSE) {
                if (objsense_section_read) {
                    std::cerr << "Error: More than one OBJSENSE section\n";
                    goto error;
                }
                objsense_section_read = true;
                current_section = SECTION_OBJSENSE;
            } else if (sec == SECTION_ROWS) {
                if (rows_section_read) {
                    std::cerr << "Error: More than one ROWS section\n";
                    goto error;
                }
                rows_section_read = true;
                current_section = SECTION_ROWS;
            } else if (sec == SECTION_COLUMNS) {
                if (columns_section_read) {
                    std::cerr << "Error: More than one COLUMNS section\n";
                    goto error;
                }
                if (!rows_section_read) {
                    std::cerr << "Error: ROWS section must come before COLUMNS\n";
                    goto error;
                }
                columns_section_read = true;
                current_section = SECTION_COLUMNS;
            } else if (sec == SECTION_RHS) {
                if (rhs_section_read) {
                    std::cerr << "Error: More than one RHS section\n";
                    goto error;
                }
                if (!rows_section_read || !columns_section_read) {
                    std::cerr << "Error: RHS section must come after ROWS and COLUMNS\n";
                    goto error;
                }
                rhs_section_read = true;
                current_section = SECTION_RHS;
            } else if (sec == SECTION_BOUNDS) {
                if (bounds_section_read) {
                    std::cerr << "Error: More than one BOUNDS section\n";
                    goto error;
                }
                if (!columns_section_read) {
                    std::cerr << "Error: BOUNDS section must come after COLUMNS\n";
                    goto error;
                }
                bounds_section_read = true;
                current_section = SECTION_BOUNDS;
            } else if (sec == SECTION_RANGES) {
                if (ranges_section_read) {
                    std::cerr << "Error: More than one RANGES section\n";
                    goto error;
                }
                if (!rows_section_read || !columns_section_read) {
                    std::cerr << "Error: RANGES section must come after ROWS and COLUMNS\n";
                    goto error;
                }
                ranges_section_read = true;
                current_section = SECTION_RANGES;
            } else if (sec == SECTION_QUADOBJ) {
                if (quadobj_section_read) {
                    std::cerr << "Error: More than one QUADOBJ section\n";
                    goto error;
                }
                if (!columns_section_read) {
                    std::cerr << "Error: QUADOBJ section must come after COLUMNS\n";
                    goto error;
                }
                quadobj_section_read = true;
                current_section = SECTION_QUADOBJ;
            } else if (sec == SECTION_ENDATA) {
                if (endata_read) {
                    std::cerr << "Error: More than one ENDATA section\n";
                    goto error;
                }
                endata_read = true;
                break;
            }
            
            continue;
        }
        
        /* Process line based on current section */
        if (current_section == SECTION_OBJSENSE) {
            read_objsense_line(qps, &card);
        } else if (current_section == SECTION_ROWS) {
            read_rows_line(qps, &card);
        } else if (current_section == SECTION_COLUMNS) {
            /* Check for integer markers */
            if (strcmp(card.f2, "'MARKER'") == 0) {
                if (strcmp(card.f3, "'INTORG'") == 0) {
                    integer_section = true;
                } else if (strcmp(card.f3, "'INTEND'") == 0) {
                    integer_section = false;
                } else {
                    std::cerr << "Error: Ignoring marker " << card.f3 << " at line " << card.nline << "\n";
                }
                continue;
            }
            read_columns_line(qps, &card, integer_section);
        } else if (current_section == SECTION_RHS) {
            read_rhs_line(qps, &card);
        } else if (current_section == SECTION_BOUNDS) {
            read_bounds_line(qps, &card);
        } else if (current_section == SECTION_RANGES) {
            read_ranges_line(qps, &card);
        } else if (current_section == SECTION_QUADOBJ) {
            read_quadobj_line(qps, &card);
        }
    }
    
    if (!endata_read) {
        std::cerr << "Warning: Reached end of file before ENDATA section\n";
    }
    
    /* Free I/O buffer */
    if (buffer) free(buffer);
    
    /* Finalize variable bounds */
    for (int j = 0; j < qps->nvar; j++) {
        HPRLP_FLOAT l = qps->lvar[j];
        HPRLP_FLOAT u = qps->uvar[j];
        VariableType vt = qps->vartypes[j];
        
        if (isnan(l) && isnan(u)) {
            /* No bounds specified, use defaults */
            qps->lvar[j] = 0.0;
            qps->uvar[j] = (vt == VTYPE_MARKED) ? 1.0 : INFINITY;
        } else if (isnan(l) && !isnan(u)) {
            /* Only upper bound specified */
            if (u < 0) {
                qps->lvar[j] = -INFINITY;
            } else {
                qps->lvar[j] = 0.0;
            }
        } else if (!isnan(l) && isnan(u)) {
            /* Only lower bound specified */
            qps->uvar[j] = INFINITY;
        }
        
        /* Convert marked variables to integer */
        if (vt == VTYPE_MARKED) {
            qps->vartypes[j] = VTYPE_INTEGER;
        }
    }
    
    return qps;
    
error:
    if (buffer) free(buffer);
    qpsdata_free(qps);
    return NULL;
}

QPSData* readqps(const char *filename, MPSFormat format) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        std::cerr << "Error: Cannot open file " << filename << "\n";
        return NULL;
    }
    
    QPSData *qps = readqps_from_file(fp, format);
    fclose(fp);
    
    return qps;
}

/* ============================================================================
 * CSR Matrix Functions
 * ========================================================================== */

/* Create a CSR matrix structure */
CSRMatrix* csr_create(int nrows, int ncols, int nnz) {
    CSRMatrix *csr = (CSRMatrix*)malloc(sizeof(CSRMatrix));
    if (!csr) return NULL;
    
    csr->nrows = nrows;
    csr->ncols = ncols;
    csr->nnz = nnz;
    
    csr->row_ptr = (int*)malloc((nrows + 1) * sizeof(int));
    csr->col_idx = (int*)malloc(nnz * sizeof(int));
    csr->values = (HPRLP_FLOAT*)malloc(nnz * sizeof(HPRLP_FLOAT));
    
    if (!csr->row_ptr || !csr->col_idx || !csr->values) {
        csr_free(csr);
        return NULL;
    }
    
    return csr;
}

/* Free a CSR matrix */
void csr_free(CSRMatrix *csr) {
    if (!csr) return;
    free(csr->row_ptr);
    free(csr->col_idx);
    free(csr->values);
    free(csr);
}

/* Comparison function for qsort (sort by row, then column) */
static int compare_coo_entries(const void *a, const void *b) {
    typedef struct { int row; int col; HPRLP_FLOAT val; } COOEntry;
    const COOEntry *ea = (const COOEntry *)a;
    const COOEntry *eb = (const COOEntry *)b;
    
    if (ea->row != eb->row) {
        return ea->row - eb->row;
    }
    return ea->col - eb->col;
}

/* Convert COO (Coordinate) format to CSR (Compressed Sparse Row) format
 * 
 * Parameters:
 *   nrows: Number of rows in the matrix
 *   ncols: Number of columns in the matrix
 *   nnz: Number of nonzero entries
 *   row_indices: Array of row indices (size: nnz)
 *   col_indices: Array of column indices (size: nnz)
 *   values: Array of values (size: nnz)
 *
 * Returns:
 *   Pointer to newly allocated CSRMatrix, or NULL on failure
 *
 * Note: This function sorts the input data and handles duplicate entries
 *       by summing their values (as per standard sparse matrix conventions)
 */
CSRMatrix* coo_to_csr(int nrows, int ncols, int nnz, 
                      const int *row_indices, const int *col_indices, 
                      const HPRLP_FLOAT *values) {
    if (nnz == 0) {
        /* Create empty matrix */
        CSRMatrix *csr = csr_create(nrows, ncols, 0);
        if (csr) {
            for (int i = 0; i <= nrows; i++) {
                csr->row_ptr[i] = 0;
            }
        }
        return csr;
    }
    
    /* Create temporary array for sorting */
    typedef struct { int row; int col; HPRLP_FLOAT val; } COOEntry;
    COOEntry *entries = (COOEntry*)malloc(nnz * sizeof(COOEntry));
    if (!entries) return NULL;
    
    /* Copy data to temporary array */
    for (int i = 0; i < nnz; i++) {
        entries[i].row = row_indices[i];
        entries[i].col = col_indices[i];
        entries[i].val = values[i];
    }
    
    /* Sort by row, then by column */
    qsort(entries, nnz, sizeof(COOEntry), compare_coo_entries);
    
    /* Count unique entries and sum duplicates */
    int unique_nnz = 1;
    for (int i = 1; i < nnz; i++) {
        if (entries[i].row != entries[i-1].row || 
            entries[i].col != entries[i-1].col) {
            unique_nnz++;
        }
    }
    
    /* Create CSR matrix */
    CSRMatrix *csr = csr_create(nrows, ncols, unique_nnz);
    if (!csr) {
        free(entries);
        return NULL;
    }
    
    /* Build CSR structure */
    int csr_idx = 0;
    
    /* Initialize row_ptr */
    for (int i = 0; i <= nrows; i++) {
        csr->row_ptr[i] = 0;
    }
    
    /* Fill in the first entry */
    csr->col_idx[0] = entries[0].col;
    csr->values[0] = entries[0].val;
    csr_idx = 1;
    
    /* Process remaining entries */
    for (int i = 1; i < nnz; i++) {
        if (entries[i].row == entries[i-1].row && 
            entries[i].col == entries[i-1].col) {
            /* Duplicate entry - sum the values */
            csr->values[csr_idx - 1] += entries[i].val;
        } else {
            /* New entry */
            csr->col_idx[csr_idx] = entries[i].col;
            csr->values[csr_idx] = entries[i].val;
            csr_idx++;
        }
    }
    
    /* Build row_ptr array */
    int row = 0;
    csr->row_ptr[0] = 0;
    
    for (int i = 0; i < unique_nnz; i++) {
        /* Find which row this entry belongs to */
        int entry_row = entries[i].row;
        
        /* Fill row_ptr for all rows up to and including this entry's row */
        while (row < entry_row) {
            row++;
            csr->row_ptr[row] = i;
        }
    }
    
    /* Fill remaining row pointers */
    while (row < nrows) {
        row++;
        csr->row_ptr[row] = unique_nnz;
    }
    
    free(entries);
    return csr;
}

/* Get the constraint matrix A in CSR format from QPSData
 *
 * This is a convenience function that converts the COO format
 * matrix stored in QPSData to CSR format.
 *
 * Returns:
 *   Pointer to newly allocated CSRMatrix, or NULL on failure
 */
CSRMatrix* qpsdata_get_csr_matrix(const QPSData *qps) {
    if (!qps) return NULL;
    
    return coo_to_csr(qps->ncon, qps->nvar, qps->annz,
                      qps->arows, qps->acols, qps->avals);
}

/* 
 * Formulation function from arrays - takes matrix and bounds arrays
 * and creates LP_info_cpu structure after filtering empty/unconstrained rows
 * 
 * This function is used by both solve_lp and solve_mps_file to ensure
 * consistent row deletion logic.
 * 
 * @param csr_A CSR matrix for constraint matrix A
 * @param AL Lower bounds for constraints (size: m)
 * @param AU Upper bounds for constraints (size: m)
 * @param l Lower bounds for variables (size: n)
 * @param u Upper bounds for variables (size: n)
 * @param c Objective coefficients (size: n)
 * @param obj_constant Objective constant term
 * @param lp Output LP_info_cpu structure
 */
/**
 * Direct model construction WITHOUT preprocessing
 */
void build_model_from_arrays(const CSRMatrix *csr_A,
                             const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                             const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                             const HPRLP_FLOAT *c,
                             HPRLP_FLOAT obj_constant,
                             LP_info_cpu *lp) {
    if (!csr_A || !AL || !AU || !l || !u || !c || !lp) {
        std::cerr << "Error: Null pointer in build_model_from_arrays\n";
        return;
    }
    
    int m = csr_A->nrows;
    int n = csr_A->ncols;
    int nnz = csr_A->nnz;
    
    /* Validate dimensions */
    if (m <= 0 || n <= 0 || nnz <= 0) {
        std::cerr << "Error: Invalid dimensions in build_model_from_arrays: m=" << m << ", n=" << n << ", nnz=" << nnz << std::endl;
        return;
    }
    
    /* Validate CSR structure */
    if (csr_A->row_ptr[0] != 0) {
        std::cerr << "Error: Invalid CSR format: row_ptr[0] = " << csr_A->row_ptr[0] << ", expected 0\n";
        return;
    }
    if (csr_A->row_ptr[m] != nnz) {
        std::cerr << "Error: Invalid CSR format: row_ptr[" << m << "] = " << csr_A->row_ptr[m] << ", expected " << nnz << "\n";
        return;
    }
    
    std::cout << "problem information: nRow = " << m << ", nCol = " << n << ", nnz A = " << nnz << std::endl;
    std::cout << "                     number of equalities = 0" << std::endl;
    std::cout << "                     number of inequalities = " << m << std::endl;
    std::cout << std::endl;
    
    /* Set dimensions */
    lp->m = m;
    lp->n = n;
    
    /* Allocate and populate constraint matrix A */
    lp->A = (sparseMatrix*)malloc(sizeof(sparseMatrix));
    lp->A->row = m;
    lp->A->col = n;
    lp->A->numElements = nnz;
    lp->A->rowPtr = (int*)malloc((m + 1) * sizeof(int));
    lp->A->colIndex = (int*)malloc(nnz * sizeof(int));
    lp->A->value = (HPRLP_FLOAT*)malloc(nnz * sizeof(HPRLP_FLOAT));
    
    memcpy(lp->A->rowPtr, csr_A->row_ptr, (m + 1) * sizeof(int));
    memcpy(lp->A->colIndex, csr_A->col_idx, nnz * sizeof(int));
    memcpy(lp->A->value, csr_A->values, nnz * sizeof(HPRLP_FLOAT));
    
    /* Allocate and copy constraint bounds */
    lp->AL = (HPRLP_FLOAT*)malloc(m * sizeof(HPRLP_FLOAT));
    lp->AU = (HPRLP_FLOAT*)malloc(m * sizeof(HPRLP_FLOAT));
    memcpy(lp->AL, AL, sizeof(HPRLP_FLOAT) * m);
    memcpy(lp->AU, AU, sizeof(HPRLP_FLOAT) * m);
    
    /* Allocate and copy objective coefficients */
    lp->c = (HPRLP_FLOAT*)malloc(n * sizeof(HPRLP_FLOAT));
    memcpy(lp->c, c, sizeof(HPRLP_FLOAT) * n);
    
    /* Allocate and copy variable bounds */
    lp->l = (HPRLP_FLOAT*)malloc(n * sizeof(HPRLP_FLOAT));
    lp->u = (HPRLP_FLOAT*)malloc(n * sizeof(HPRLP_FLOAT));
    memcpy(lp->l, l, sizeof(HPRLP_FLOAT) * n);
    memcpy(lp->u, u, sizeof(HPRLP_FLOAT) * n);
    
    /* Set objective constant - AT will be generated later in copy_lpinfo_to_device */
    lp->obj_constant = obj_constant;
}

void build_model_from_mps(const char* mps_fp, LP_info_cpu *lp) {
    std::cout << "Start reading file....\n";
    
    /* Read MPS file using readqps */
    clock_t start_time = clock();
    QPSData *qps = readqps(mps_fp, MPS_FREE);  /* Try free format first */
    clock_t end_time = clock();
    HPRLP_FLOAT read_time = ((HPRLP_FLOAT)(end_time - start_time)) / CLOCKS_PER_SEC;
    std::cout << "File reading time: " << std::fixed << std::setprecision(4) << read_time << " seconds\n" << std::defaultfloat;
    if (!qps) {
        std::cerr << "Error: Failed to read MPS file\n";
        return;
    }
    
    /* Convert COO matrix to CSR format */
    CSRMatrix *csr_A = qpsdata_get_csr_matrix(qps);
    if (!csr_A) {
        std::cerr << "Error: Failed to convert matrix to CSR format\n";
        qpsdata_free(qps);
        return;
    }
    
    /* Call build_model_from_arrays for direct construction */
    build_model_from_arrays(csr_A, qps->lcon, qps->ucon, 
                            qps->lvar, qps->uvar, qps->c, 
                            qps->c0, lp);
    
    /* Free CSR matrix and QPSData */
    csr_free(csr_A);
    qpsdata_free(qps);
}