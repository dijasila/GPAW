#include <Python.h>

typedef struct
{
  const double* A_gm;  // function values
  int nm;              // number of functions (2*l+1)
  int M;               // global number of first function
  int W;               // volume number
} LFVolume;


typedef struct 
{
  PyObject_HEAD
  double dv;                 // volume per grid point
  int nW;                    // number of volumes
  int nB;                    // number of boundary points
  double* work_gm;           // work space
  LFVolume* volume_W;        // pointers to volumes
  LFVolume* volume_i;        // pointers to volumes at current grid point
  int* G_B;                  // boundary grid points
  int* W_B;                  // volume numbers
  int* i_W;                  // mapping from all volumes to current volumes
  int* ng_W;                 // number of grid poinst per volume
  bool gamma;                // Gamma-point calculation?
  complex double* phase_kW;  // phase factors: exp(ik.R)
  complex double* phase_i;   // phase factors for current volumes
} LFCObject;
