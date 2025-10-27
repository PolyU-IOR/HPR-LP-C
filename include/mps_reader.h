#ifndef HPRLP_MPS_READER_H
#define HPRLP_MPS_READER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "structs.h"

/**
 * MPS File Reader for C++
 * 
 * This implementation provides functionality to read MPS (Mathematical Programming System)
 * format files and convert them to internal LP data structures.
 * 
 * The implementation is based on and follows the design of qpsreader.jl but adapted
 * for MPS format, which is the standard format for linear programming problems.
 * 
 * Note: While the internal structures use "QPS" naming for historical reasons,
 * this reader specifically handles MPS format files.
 * 
 * References: QPSReader.jl, HiGHS
 */

/* Variable types */
enum VariableType {
    VTYPE_CONTINUOUS,
    VTYPE_BINARY,
    VTYPE_INTEGER,
    VTYPE_MARKED,  /* Internal use for integer variables */
    VTYPE_SEMICONTINUOUS,
    VTYPE_SEMIINTEGER
};

/* Row types (constraint types) */
enum RowType {
    RTYPE_OBJECTIVE,    /* 'N' type row */
    RTYPE_EQUALTO,      /* 'E' type row */
    RTYPE_LESSTHAN,     /* 'L' type row */
    RTYPE_GREATERTHAN   /* 'G' type row */
};

/* CSR (Compressed Sparse Row) matrix structure */
struct CSRMatrix {
    int nrows;          /* Number of rows */
    int ncols;          /* Number of columns */
    int nnz;            /* Number of nonzeros */
    int *row_ptr;       /* Row pointers (size: nrows+1) */
    int *col_idx;       /* Column indices (size: nnz) */
    HPRLP_FLOAT *values;     /* Values (size: nnz) */
};

/* Objective sense */
enum MPSObjSense {
    OBJSENSE_MIN,
    OBJSENSE_MAX,
    OBJSENSE_NOTSET
};

/* MPS format type */
enum MPSFormat {
    MPS_FIXED,
    MPS_FREE
};

/* Section identifiers */
enum SectionType {
    SECTION_NONE = -1,
    SECTION_ENDATA = 0,
    SECTION_NAME = 1,
    SECTION_OBJSENSE = 2,
    SECTION_ROWS = 3,
    SECTION_COLUMNS = 4,
    SECTION_RHS = 5,
    SECTION_BOUNDS = 6,
    SECTION_RANGES = 7,
    SECTION_QUADOBJ = 8,
    SECTION_QMATRIX = 8,  /* Same as QUADOBJ */
    SECTION_OBJECT_BOUND = 10
};

/* Name-Index mapping structure with hash table */
struct NameIndexPair {
    char *name;
    int index;
    int next;  /* For chaining in hash table */
};

/* Hash table for name-index mapping */
struct NameIndexMap {
    NameIndexPair *pairs;  /* Array of all entries */
    int *buckets;          /* Hash table buckets (indices into pairs array) */
    int size;              /* Number of entries */
    int capacity;          /* Capacity of pairs array */
    int num_buckets;       /* Number of hash buckets */
};

/* QPSData structure to hold the problem data */
struct QPSData {
    /* Problem dimensions */
    int nvar;  /* Number of variables */
    int ncon;  /* Number of constraints */
    
    /* Objective function */
    MPSObjSense objsense;  /* Objective sense: minimize or maximize */
    HPRLP_FLOAT c0;          /* Constant term in objective */
    HPRLP_FLOAT *c;          /* Linear coefficients in objective (size: nvar) */
    
    /* Quadratic objective matrix Q in COO format */
    int qnnz;           /* Number of nonzeros in Q */
    int qnnz_capacity;  /* Allocated capacity */
    int *qrows;         /* Row indices */
    int *qcols;         /* Column indices */
    HPRLP_FLOAT *qvals;      /* Values */
    
    /* Constraint matrix A in COO format */
    int annz;           /* Number of nonzeros in A */
    int annz_capacity;  /* Allocated capacity */
    int *arows;         /* Row indices */
    int *acols;         /* Column indices */
    HPRLP_FLOAT *avals;      /* Values */
    
    /* Bounds */
    HPRLP_FLOAT *lcon;  /* Constraint lower bounds (size: ncon) */
    HPRLP_FLOAT *ucon;  /* Constraint upper bounds (size: ncon) */
    HPRLP_FLOAT *lvar;  /* Variable lower bounds (size: nvar) */
    HPRLP_FLOAT *uvar;  /* Variable upper bounds (size: nvar) */
    
    /* Names */
    char *name;      /* Problem name */
    char *objname;   /* Objective function name */
    char *rhsname;   /* RHS name */
    char *bndname;   /* BOUNDS name */
    char *rngname;   /* RANGES name */
    
    char **varnames;  /* Variable names (size: nvar) */
    char **connames;  /* Constraint names (size: ncon) */
    
    /* Name-index mappings */
    NameIndexMap varindices;  /* Variable name -> index */
    NameIndexMap conindices;  /* Constraint name -> index */
    
    /* Variable and constraint types */
    VariableType *vartypes;  /* Variable types (size: nvar) */
    RowType *contypes;       /* Constraint types (size: ncon) */
    
    /* Capacities for dynamic arrays */
    int var_capacity;
    int con_capacity;
};

/* MPS Card structure for parsing */
struct MPSCard {
    int nline;          /* Line number */
    bool iscomment;     /* Is this line a comment? */
    bool isheader;      /* Is this line a section header? */
    int nfields;        /* Number of fields read */
    
    /* Fields */
    char f1[256];
    char f2[256];
    char f3[256];
    char f4[256];
    char f5[256];
    char f6[256];
};

/* Function declarations */

/* Initialize and free QPSData */
QPSData* qpsdata_create();
QPSData* qpsdata_create_with_capacity(int var_cap, int con_cap, int nnz_cap);
void qpsdata_free(QPSData *qps);

/* Main reading function */
QPSData* readqps(const char *filename, MPSFormat format);
QPSData* readqps_from_file(std::FILE *fp, MPSFormat format);

/* Name-index map functions */
void namemap_init(NameIndexMap *map);
void namemap_free(NameIndexMap *map);
int namemap_get(NameIndexMap *map, const char *name, int default_value);
void namemap_set(NameIndexMap *map, const char *name, int index);

/* Utility functions */
void trim(char *str);
char* strdup_safe(const char *str);

/* CSR matrix functions */
CSRMatrix* csr_create(int nrows, int ncols, int nnz);
void csr_free(CSRMatrix *csr);
CSRMatrix* coo_to_csr(int nrows, int ncols, int nnz, 
                      const int *row_indices, const int *col_indices, 
                      const HPRLP_FLOAT *values);
CSRMatrix* qpsdata_get_csr_matrix(const QPSData *qps);

/**
 * Build model from arrays - creates LP_info_cpu structure from constraint matrix and bounds.
 * 
 * This function takes constraint matrix A (in CSR format) and bound arrays, then:
 * 1. Copies all data directly into LP_info_cpu structure
 * 2. Simply builds the model as-is from input arrays
 * 3. Does NOT perform any preprocessing or row deletion
 * 
 * @param csr_A Constraint matrix A in CSR format (not modified)
 * @param AL Lower bounds for constraints (size: m = csr_A->nrows)
 * @param AU Upper bounds for constraints (size: m = csr_A->nrows)
 * @param l Lower bounds for variables (size: n = csr_A->ncols)
 * @param u Upper bounds for variables (size: n = csr_A->ncols)
 * @param c Objective coefficients (size: n = csr_A->ncols)
 * @param obj_constant Objective constant term
 * @param lp Output LP_info_cpu structure (will be populated by this function)
 * 
 * @note The caller is responsible for freeing the LP_info_cpu structure using free_lp_info_cpu()
 * @see build_model_from_mps(), create_model_from_arrays()
 */
void build_model_from_arrays(const CSRMatrix *csr_A,
                             const HPRLP_FLOAT *AL, const HPRLP_FLOAT *AU,
                             const HPRLP_FLOAT *l, const HPRLP_FLOAT *u,
                             const HPRLP_FLOAT *c,
                             HPRLP_FLOAT obj_constant,
                             LP_info_cpu *lp);

/**
 * Build model from MPS file - creates LP_info_cpu structure from MPS file.
 * 
 * This function:
 * 1. Reads the MPS file using readqps()
 * 2. Converts the constraint matrix to CSR format
 * 3. Calls build_model_from_arrays() to create the LP_info_cpu structure
 * 
 * @param mps_fp Path to MPS file
 * @param lp Output LP_info_cpu structure (will be populated by this function)
 * 
 * @note The caller is responsible for freeing the LP_info_cpu structure using free_lp_info_cpu()
 * @see build_model_from_arrays()
 */
void build_model_from_mps(const char* mps_fp, LP_info_cpu *lp);

#endif /* HPRLP_MPS_READER_H */
