#include <Python.h>
#ifdef GPAW_WITH_SL
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include <mpi.h>
#include "extensions.h"
#include <structmember.h>
#include "mympi.h"

// BLACS
#ifdef GPAW_AIX
#define   Cblacs_gridinfo_ Cblacs_gridinfo
#define   Cblacs_gridinit_ Cblacs_gridinit
#define   Cblacs_pinfo_    Cblacs_pinfo
#define   Csys2blacs_handle_ Csys2blacs_handle
#endif

void Cblacs_gridinfo_(int ConTxt, int *nprow, int *npcol,
              int *myrow, int *mycol);

void Cblacs_gridinit_(int *ConTxt, char* order, int nprow, int npcol);

void Cblacs_pinfo_(int *mypnum, int *nprocs);

int Csys2blacs_handle_(MPI_Comm SysCtxt);
// End of BLACS

// ScaLAPACK
#ifdef GPAW_AIX
#define   descinit_  descinit
#define   numroc_    numroc
#define   Cpdgemr2d_ Cpdgemr2d
#define   pdlamch_   pdlamch

#define   pdpotrf_  pdpotrf
#define   pdpotri_  pdpotri
#define   pzpotrf_  pzpotrf
#define   pzpotri_  pzpotri
#define   pdtrtri_  pdtrtri
#define   pztrtri_  pztrtri

#define   pdsyevd_  pdsyevd
#define   pdsyev_   pdsyev
#define   pdsyevx_  pdsyevx
#define   pdsygvx_  pdsygvx
#define   pzheev_   pzheev
#endif

// tools
void descinit_(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc,
               int* icsrc, int* ictxt,
               int* lld, int* info);

int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

void Cpdgemr2d_(int m, int n, double *A, int IA, int JA, int *descA,
               double *B, int IB, int JB, int *descB, int gcontext);

double pdlamch_(int* ictxt, char* cmach);

// cholesky
void pdpotrf_(char *uplo, int *n, double* a, int *ia, int* ja, int* desca,
              int *info);
void pdpotri_(char *uplo, int *n, double* a, int *ia, int* ja, int* desca,
              int *info);

void pzpotrf_(char *uplo, int *n, void* a, int *ia, int* ja, int* desca,
              int *info);
void pzpotri_(char *uplo, int *n, void* a, int *ia, int* ja, int* desca,
              int *info);


void pdtrtri_(char *uplo, char *diag, int *n, double* a, int *ia, int* ja,
              int* desca, int *info);
void pztrtri_(char *uplo, char *diag, int *n, void* a, int *ia, int* ja,
              int* desca, int *info);

// diagonalization
void pdsyevd_(char *jobz, char *uplo, int *n, double* a, int *ia, int* ja,
              int* desca, double *w,
              double* z, int *iz, int* jz, int* descz, double *work,
              int *lwork, int *iwork, int *liwork, int *info);

void pdsyev_(char *jobz, char *uplo, int *n, double* a, int *ia, int* ja,
             int* desca, double *w,
             double* z, int *iz, int* jz, int* descz, double *work,
             int *lwork, int *info);

void pdsyevx_(char *jobz, char *range, char *uplo, int *n, double* a,
              int *ia, int* ja, int* desca, double* vl,
              double* vu, int* il, int* iu, double* abstol, int* m, int* nz,
              double* w, double* orfac, double* z, int *iz,
              int* jz, int* descz, double *work, int *lwork, int *iwork,
              int *liwork, int *ifail, int *iclustr, double* gap, int *info);

void pdsygvx_(int *ibtype, char *jobz, char *range, char *uplo, int *n,
              double* a, int *ia, int* ja,
              int* desca, double* b, int *ib, int* jb, int* descb,
              double* vl, double* vu, int* il, int* iu,
              double* abstol, int* m, int* nz, double* w, double* orfac,
              double* z, int *iz, int* jz, int* descz,
              double *work, int *lwork, int *iwork, int *liwork, int *ifail,
              int *iclustr, double* gap, int *info);

void pzheev_(char *jobz, char *uplo, int *n, double* a, int *ia, int* ja,
             int* desca, double *w, double* z, int *iz, int* jz,
             int* descz, double *work, int *lwork, double *rwork,
             int *lrwork, int *info);


PyObject* blacs_array(PyObject *self, PyObject *args)
{
  PyObject*  comm_obj;     // communicator
  char order='R';
  int m, n, nprow, npcol, mb, nb, lld;
  int nprocs;
  int ConTxt = -1;
  int iam = 0;
  int rsrc = 0;
  int csrc = 0;
  int myrow = -1;
  int mycol = -1;
  int desc[9];

  npy_intp desc_dims[1] = {9};
  PyArrayObject* desc_obj = (PyArrayObject*)PyArray_SimpleNew(1, desc_dims, NPY_INT);

  if (!PyArg_ParseTuple(args, "Oiiiiii", &comm_obj, &m, &n, &nprow, &npcol, &mb, &nb))
    return NULL;

  if (comm_obj == Py_None)
    {
      desc[0] = 1;  // BLOCK_CYCLIC_2D
      desc[1] = -1; // Tells BLACS to ignore me.
      desc[2] = m;
      desc[3] = n;
      desc[4] = mb;
      desc[5] = nb;
      desc[6] = 0;
      desc[7] = 0;
      desc[8] = 1;
    }
  else
    {
      // Create blacs grid on this communicator
      MPI_Comm comm = ((MPIObject*)comm_obj)->comm;

      // Get my id and nprocs. This is for debugging purposes only
      Cblacs_pinfo_(&iam, &nprocs);
      MPI_Comm_size(comm, &nprocs);
      // printf("iam=%d,nprocs=%d\n",iam,nprocs);

      // Create blacs grid on this communicator continued
      ConTxt = Csys2blacs_handle(comm);
      Cblacs_gridinit_(&ConTxt, &order, nprow, npcol);
      // printf("ConTxt=%d,nprow=%d,npcol=%d\n",ConTxt,nprow,npcol);
      Cblacs_gridinfo_(ConTxt, &nprow, &npcol, &myrow, &mycol);

      lld = numroc_(&m, &mb, &myrow, &rsrc, &nprow);

      desc[0] = 1; // BLOCK_CYCLIC_2D
      desc[1] = ConTxt;
      desc[2] = m;
      desc[3] = n;
      desc[4] = mb;
      desc[5] = nb;
      desc[6] = 0;
      desc[7] = 0;
      desc[8] = MAX(1, lld);
    }
  memcpy(desc_obj->data, desc, 9*sizeof(int));

  return Py_BuildValue("O",desc_obj);
}

PyObject* blacs_redist(PyObject *self, PyObject *args)
{
    PyArrayObject* a_obj; //source matrix
    PyArrayObject* adesc; //source descriptor
    PyArrayObject* bdesc; //destination descriptor
    int m = 0;
    int n = 0;
    int ConTxt;
    static int one = 1;

    if (!PyArg_ParseTuple(args, "OOO|ii", &a_obj, &adesc, &bdesc, &m, &n))
      return NULL;

    // adesc
    int a_mycol = -1;
    int a_myrow = -1;
    int a_nprow, a_npcol; 
    int a_ConTxt = INTP(adesc)[1];
    int a_m = INTP(adesc)[2];
    int a_n = INTP(adesc)[3];
    int a_mb = INTP(adesc)[4];
    int a_nb = INTP(adesc)[5];
    int a_rsrc = INTP(adesc)[6];
    int a_csrc = INTP(adesc)[7];

    // If m and n not specified, redistribute all rows and columns of a.
    if ((m == 0) | (n == 0))
      {
        m = a_m;
        n = a_n;	  
      }

    // bdesc
    int b_mycol = -1;
    int b_myrow = -1;
    int b_nprow, b_npcol; 
    int b_ConTxt = INTP(bdesc)[1];
    int b_m = INTP(bdesc)[2];
    int b_n = INTP(bdesc)[3];
    int b_mb = INTP(bdesc)[4];
    int b_nb = INTP(bdesc)[5];
    int b_rsrc = INTP(bdesc)[6];
    int b_csrc = INTP(bdesc)[7];

    // Get adesc and bdesc grid info
    Cblacs_gridinfo_(a_ConTxt, &a_nprow, &a_npcol,&a_myrow, &a_mycol);

    Cblacs_gridinfo_(b_ConTxt, &b_nprow, &b_npcol,&b_myrow, &b_mycol);
    // printf("b_ConTxt=%d,b_nprow=%d,b_npcol=%d,b_myrow=%d,b_mycol=%d\n",b_ConTxt,b_nprow,b_npcol,b_myrow,b_mycol);

    int b_locM = numroc_(&b_m, &b_mb, &b_myrow, &b_rsrc, &b_nprow);
    int b_locN = numroc_(&b_n, &b_nb, &b_mycol, &b_csrc, &b_npcol);
    
    if (b_locM < 0) b_locM = 0;
    if (b_locN < 0) b_locN = 0;

    npy_intp b_dims[2] = {b_locM, b_locN};
    PyArrayObject* b_obj = (PyArrayObject*)PyArray_SimpleNew(2, b_dims, NPY_DOUBLE);


    // Determine the largest grid because the ConTxt that is passed
    // to Cpdgemr2d must encompass both grids (I think). The SCALAPACK
    // documentation is not clear on this point. 
    printf("a_nprow=%d,a_npcol=%d,b_nprow=%d,b_npcol=%d\n",a_nprow,a_npcol,b_nprow,b_npcol);
    if ((a_nprow*a_npcol) > (b_nprow*b_npcol))
      {
        ConTxt = a_ConTxt;
      }
    else
      {
	ConTxt = b_ConTxt;
      }


    // It appears that the memory requirements for Cpdgemr2d are non-trivial.
    // Consideer A_loc, B_loc to be the local piece of the global array. Then
    // to perform this operation you will need an extra A_loc, B_loc worth of
    // memory. Hence, for --state-parallelization=4 the memory savings are
    // about nbands-by-nbands is exactly 1/4*(nbands-by-nbands). For --state-
    // parallelization=8 it is about 3/4*(nbands-by-nbands). For --state-
    // parallelization=B it is about (B-2)/B*(nbands-by-nbands).
    if (ConTxt != -1)
      {
	Cpdgemr2d_(m, n, DOUBLEP(a_obj), one, one, INTP(adesc), DOUBLEP(b_obj), one, one, INTP(bdesc), ConTxt);
	return Py_BuildValue("O",b_obj);
      }
    else
      {
	Py_RETURN_NONE;
      }
}
PyObject* scalapack_diagonalize_dc(PyObject *self, PyObject *args)
{
    // Standard Driver for Divide and Conquer Algorithm
    // Computes all eigenvalues and eigenvectors
 
    PyArrayObject* a_obj; // symmetric matrix
    PyArrayObject* adesc; // symmetric matrix description vector
    int z_mycol = -1;
    int z_myrow = -1;
    int z_nprow, z_npcol;
    int z_type, z_ConTxt, z_m, z_n, z_mb, z_nb, z_rsrc, z_csrc;
    int zdesc[9];
    char jobz = 'V'; // eigenvectors also
    char uplo = 'U'; // work with upper
    static int one = 1;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &adesc))
      return NULL;

    // adesc
    int a_type  = INTP(adesc)[0];
    int a_ConTxt = INTP(adesc)[1];
    int a_m = INTP(adesc)[2];
    int a_n = INTP(adesc)[3];
    int a_mb = INTP(adesc)[4];
    int a_nb = INTP(adesc)[5];
    int a_rsrc = INTP(adesc)[6];
    int a_csrc = INTP(adesc)[7];

    // Note that A is symmetric, so n = a_m = a_n;
    // We do not test for that here.
    int n = a_n;
    if (a_ConTxt == -1) n = 0; // Eigenvalues end up on same tasks
                               // as eigenvectors


    // zdesc = adesc
    // This is generally not required, as long as the 
    // alignment properties are satisfied, see pdsyevd.f
    // In the context of GPAW, don't see why zdesc would
    // not be equal to adesc so I am just hard-coding it in.
    z_type   = a_type;
    z_ConTxt = a_ConTxt;
    z_m      = a_m;
    z_n      = a_n;
    z_mb     = a_mb;
    z_nb     = a_nb;
    z_rsrc   = a_rsrc;
    z_csrc   = a_csrc;
    zdesc[0] = z_type;
    zdesc[1] = z_ConTxt;
    zdesc[2] = z_m;
    zdesc[3] = z_n;
    zdesc[4] = z_mb;
    zdesc[5] = z_nb;
    zdesc[6] = z_rsrc;
    zdesc[7] = z_csrc;


    Cblacs_gridinfo_(z_ConTxt, &z_nprow, &z_npcol,&z_myrow, &z_mycol);

    int z_locM = numroc_(&z_m, &z_mb, &z_myrow, &z_rsrc, &z_nprow);
    int z_locN = numroc_(&z_n, &z_nb, &z_mycol, &z_csrc, &z_npcol);
    
    if (z_locM < 0) z_locM = 0;
    if (z_locN < 0) z_locN = 0;

    // Eigenvectors
    npy_intp z_dims[2] = {z_locM, z_locN};
    PyArrayObject* z_obj = (PyArrayObject*)PyArray_SimpleNew(2, z_dims, NPY_DOUBLE);
    
    // Eigenvalues
    npy_intp w_dims[1] = {n};
    PyArrayObject* w_obj = (PyArrayObject*)PyArray_SimpleNew(1, w_dims, NPY_DOUBLE);

    if (z_ConTxt != -1)
      {
        // Query part, need to find the optimal size of a number of work arrays
	int info;
        double* work;
        work = GPAW_MALLOC(double, 3);
        int querylwork = -1;
        int* iwork;
        iwork = GPAW_MALLOC(int, 1);
        int queryliwork = 1;
        pdsyevd_(&jobz, &uplo, &n, DOUBLEP(a_obj), &one, &one, INTP(adesc), 
                 DOUBLEP(w_obj), DOUBLEP(z_obj), &one, &one, zdesc, 
                 work, &querylwork, iwork, &queryliwork, &info);
	// Computation part
        int lwork = (int)work[0];
        free(work);
        int liwork = (int)iwork[0];
        free(iwork);
        work = GPAW_MALLOC(double, lwork);
        iwork = GPAW_MALLOC(int, liwork);
        pdsyevd_(&jobz, &uplo, &n, DOUBLEP(a_obj), &one, &one, INTP(adesc), 
                 DOUBLEP(w_obj), DOUBLEP(z_obj), &one, &one, zdesc, 
                 work, &lwork, iwork, &liwork, &info);

        free(work);
	free(iwork);
        return Py_BuildValue("(OO)", w_obj, z_obj);
      }
    else
      {
	Py_RETURN_NONE;
      }
}
#endif
