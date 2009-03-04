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

// ScaLapack
#ifdef GPAW_AIX
#define   descinit_  descinit
#define   numroc_  numroc
#define   Cpdgemr2d_  Cpdgemr2d

#define   pdpotrf_  pdpotrf
#define   pdpotri_  pdpotri
#define   pzpotrf_  pzpotrf
#define   pzpotri_  pzpotri
#define   pdtrtri_  pdtrtri
#define   pztrtri_  pztrtri

#define   pdsyevd_  pdsyevd
#define   pdsyev_  pdsyev
#define   pdsyevx_  pdsyevx
#define   pdsygvx_  pdsygvx
#define   pzheev_  pzheev
#endif

// tools
void descinit_(int* desc, int* m, int* n, int* mb, int* nb, int* irsrc,
               int* icsrc, int* ictxt,
               int* lld, int* info);

int numroc_(int* n, int* nb, int* iproc, int* isrcproc, int* nprocs);

void Cpdgemr2d_(int m, int n, double *A, int IA, int JA, int *descA,
               double *B, int IB, int JB, int *descB, int gcontext);

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

  if (((MPIObject*)comm_obj)->comm == MPI_COMM_NULL)
    {
      desc[0] = 1;
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

      desc[0] = 1;
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
    PyArrayObject* a_obj;     //source matrix
    PyArrayObject* adesc; //description vector
    PyArrayObject* bdesc; //description vector
    int m, n, ConTxt; 
    if (!PyArg_ParseTuple(args, "OOOiii", &a_obj, &adesc, &bdesc, &m, &n, &ConTxt))
      return NULL;
 
    static int one = 1;

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

    Cblacs_gridinfo_(b_ConTxt, &b_nprow, &b_npcol,&b_myrow, &b_mycol);
    // printf("b_ConTxt=%d,b_nprow=%d,b_npcol=%d,b_myrow=%d,b_mycol=%d",b_ConTxt,b_nprow,b_npcol,b_myrow,b_mycol);

    int b_locM = numroc_(&b_m, &b_mb, &b_myrow, &b_rsrc, &b_nprow);
    int b_locN = numroc_(&b_n, &b_nb, &b_mycol, &b_csrc, &b_npcol);
    
    if (b_locM < 0) b_locM = 0;
    if (b_locN < 0) b_locN = 0;

    npy_intp b_dims[2] = {b_locM, b_locN};
    PyArrayObject* b_obj = (PyArrayObject*)PyArray_SimpleNew(2, b_dims, NPY_DOUBLE);

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
#endif
