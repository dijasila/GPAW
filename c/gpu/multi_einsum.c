#include "../extensions.h"

#define GPAW_ARRAY_DISABLE_NUMPY
#define GPAW_ARRAY_ALLOW_CUPY
#include "../array.h"
#undef GPAW_ARRAY_DISABLE_NUMPY

void multi_einsum_launch_kernel(char* str,
                                int problems,  // p index
                                int arguments, // a index
                                int maxind,    // i index
                                int* dimensions_pai,
                                int* strides_pai,
                                double** array_pointers_pa,
                                int add,
                                int is_complex,
                                int is_complex_out,
                                const char** r);

PyObject* multi_einsum_gpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char *kwlist[] = {"string", "arguments", "out", "add", NULL};
    char* string;


    // arguments is expected to be list of tuples
    PyObject* arguments_obj;
    PyObject* out_obj = NULL;
    PyObject* add_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|OO", kwlist, 
                                     &string, &arguments_obj, &out_obj, &add_obj))
        return NULL;

    if ((add_obj != NULL) && (out_obj != NULL))
    {
        PyErr_SetString(PyExc_RuntimeError, "Cannot set both out and add arguments.");
        return NULL;
    }
    bool add = add_obj != NULL;
    if (add)
    {
        out_obj = add_obj;
    }

    int problems = PyList_Size(arguments_obj);
    if (problems == 0)
    {
        Py_RETURN_NONE;      
    }
    if (PyErr_Occurred())
    {
        printf("1\n");
        return NULL;
    }

    int arguments = PyTuple_Size(PyList_GetItem(arguments_obj, 0)) + 1; // We add one, because we append out
    if (PyErr_Occurred())
    {
        printf("2\n");
        return NULL;
    }

    double** array_pointers = (double**) malloc(problems * arguments * sizeof(double*));
    // max dimensions is 4
    int* dimensions = (int*) malloc(problems * arguments * 4 * sizeof(int));
    int* strides = (int*) malloc(problems * arguments * 4 * sizeof(int));
    int first_item_size = -1;
    int first_output_size = -1;
    for (int i=0; i<problems; i++)
    {
        PyObject* args = PyList_GetItem(arguments_obj, i);
        PyObject* output = PyList_GetItem(out_obj, i);
        if (PyErr_Occurred())
        {
            printf("3\n");
            goto error;
        }
        if (PyTuple_Size(args) != arguments - 1)
        {
            PyErr_Format(PyExc_RuntimeError, "Inconsistent number of arguments at problem %d.", i);
            goto error;
        }
        for (int j=0; j<arguments; j++)
        {
            PyObject* cupy_array;
            if (j < arguments - 1) 
            {
                cupy_array = PyTuple_GetItem(args, j);
            }
            else
            {
                cupy_array = output;
            }
            if (PyErr_Occurred())
            {
                printf("4\n");
                goto error;
            }
            double* array_ptr = Array_DATA(cupy_array);
            int item_size = Array_ITEMSIZE(cupy_array);
            if (j < arguments - 1)
            {
                if (first_item_size != -1)
                {
                    if (item_size != first_item_size)
                    {
                        PyErr_SetString(PyExc_RuntimeError, "All arguments must be of same dtype.");
                        goto error;
                    }
                }
                else
                {
                    first_item_size = item_size;
                }
            }
            else
            {
                if (first_output_size != -1)
                {
                    if (item_size != first_output_size)
                    {
                        PyErr_SetString(PyExc_RuntimeError, "All outputs must be of same dtype.");
                        goto error;
                    }
                }
                else
                {
                    first_output_size = item_size;
                }

            }
            if (PyErr_Occurred())
            {
                printf("5\n");
                goto error;
            }
            array_pointers[j + i * arguments] = array_ptr;

            int ndim = Array_NDIM(cupy_array);
            if (ndim > 4)
            {
                PyErr_SetString(PyExc_RuntimeError, "Arrays only up to 4 dimensions supported.");
                goto error;
            }
            int stride = 0;
            for (int k=4-1; k>=0; k--)
            {
                if (k<ndim)
                {
                    stride = Array_STRIDE(cupy_array, k) / item_size;
                }
                dimensions[k + 4*j + 4*arguments*i] = (k<ndim) ? Array_DIM(cupy_array, k) : 0;
                strides[k + 4*j + 4*arguments*i] = (k<ndim) ? stride : 0;
            }
        }
    }
    char error[50] = "";
    multi_einsum_launch_kernel(string,
                               problems, 
                               arguments,
                               4,
                               dimensions,
                               strides,
                               array_pointers,
                               add,
                               first_item_size == 16,
                               first_output_size == 16,
                               (const char**)&error);
    free(array_pointers);
    free(dimensions);
    if (strlen(error))
    {
        PyErr_SetString(PyExc_RuntimeError, error);
        return NULL;
    }
    Py_RETURN_NONE;
error:
    free(array_pointers);
    free(dimensions);
    return NULL;    
}


