#include "Python.h"


#include "../gpu.h"
#include "../gpu-complex.h"
#include "numpy/arrayobject.h"
#include "assert.h"
#include <cstdlib>

#define GPAW_ARRAY_DISABLE_NUMPY
#define GPAW_ARRAY_ALLOW_CUPY
#include "../../array.h"
#undef GPAW_ARRAY_DISABLE_NUMPY

class Tokenizer
{
    public:
        // Tokenizer will take a null terminated C-string
        Tokenizer(const char* str) : str(str), chr(str), head(0)
        {
        }

        void push()
        {
            stack[head++] = chr;
        }

        void pop()
        {
            chr = stack[--head];
        }

        void stash()
        {
            --head;
        }

        const char& peek() const
        {
            return *chr;
        }

        const char& next()
        {
            const char& c = *chr;
            if (c != 0)
            {
                chr++;
            }
            if (c == ' ')
            {
                return next();
            }
            return c;
        }

        bool parse_alphabet(char& c)
        {
            c = next();
            return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z'));
        }

        bool parse_char(char c)
        {
            char d = next();
            return d == c;
        }

    private:
        const char* str;
        const char* chr;
        const char* stack[10];
        int head;

};

class Parser
{
    public:
        Parser(Tokenizer& tokenizer) : tokenizer(tokenizer), error(""), error_set(0)
        {
        }

        void print_error()
        {
            printf("%s\n", error);
        }
        char error[50];

    protected:
        void expected_error(const char* expected_str)
        {
            if (error_set)
            {
                printf("Internal error. Error already set:");
                print_error();
            }
            error_set = true;
            char c = tokenizer.next();
            if (c != 0)
            {
                sprintf(error, "Expected %s instead of '%c'.", expected_str, c);
            }
            else
            {
                sprintf(error, "Expected %s instead of end of string.", expected_str);
            }
        }

        Tokenizer& tokenizer;


    private:
        bool error_set;

};

class EinsumArgumentParser : public Parser
{
    public:
    int indices_list[10][10];
    int indices_len[10];
    bool cc[10];
    char seen[10];
    int index_head;

    EinsumArgumentParser(Tokenizer& tokenizer) 
        : Parser(tokenizer), index_head(0), seen("")
    {
        for (int i=0;i<10;i++)
        {
            indices_len[i] = 0;
            cc[i] = false;
        }
    }

    int nindex() const
    {
        return strlen(seen);
    }

    int nindex_out() const
    {
        return indices_len[index_head-1];
    }

    int nindex_in() const
    {
        return nindex() - nindex_out();
    }


    bool is_out_index(int i) const
    {
        for (int n=0; n<indices_len[index_head-1]; n++)
        {
            if (i == indices_list[index_head-1][n])
            {
                return true;
            }
        }
        return false;
    }

    int index_in(int n) const
    {
        int index = 0;
        for (int i=0; i<nindex(); i++)
        {
            if (is_out_index(i))
            {
                continue;
            }
            if (n == index)
            {
                return i;
            }
            index ++;
        }
        return -1;
    }

    int index_out(int n)
    {
        return indices_list[index_head-1][n];
    }

    int nargs() const
    {
        return index_head;
    }

    void print() const
    {
        printf("Number of indices %d.\n", nindex());
        printf("Number of outer indices %d.\n", nindex_out());
        printf("Number of inner indices %d.\n", nindex_in());
        for (int i=0; i<index_head; i++)
        {
            if (i < index_head-1)
               printf("in: ");
            else
               printf("out: ");
            if (cc[i])
            {
                printf("*");
            }
            else
            {
                printf(" ");
            }
            for (int j=0; j<indices_len[i]; j++)
               printf("%d ", indices_list[i][j]);
            for (int j=0; j<indices_len[i]; j++)
                printf("%c ", seen[indices_list[i][j]]);
            printf("\n");
        }
    }
    
    bool parse()
    {
        bool success;
        success = parse_indices_list();
        if (!success)
        {
            return false;
        }

        success = parse_out_separator();
        if (!success)
        {
            return false;
        }
        success = parse_indices();
        if (!success)
        {
            return false;
        }
        return true;
    }


    protected:
    bool parse_out_separator()
    {
        bool success;
        tokenizer.push();
        success = tokenizer.parse_char('-');
        if (!success)
        {
            tokenizer.pop();
            expected_error("-");
            return false;
        }
        tokenizer.stash();
        tokenizer.push();
        success = tokenizer.parse_char('>');
        if (!success)
        {
            tokenizer.pop();
            expected_error("> after -");
            return false;
        }
        tokenizer.stash();
        return true;
    }
   
    bool parse_indices_list()
    {
        bool success;
        success = parse_indices();
        if (!success)
        {
            return false;
        }
        while (true)
        {
            tokenizer.push();

            success = parse_comma();
            if (!success)
            {
                tokenizer.pop();
                break;
            }
            tokenizer.stash();
            tokenizer.push();
            success = parse_indices();
            if (!success)
            {
                tokenizer.pop();
                expected_error("indices");
                return false;
            }
            tokenizer.stash();
        }
        return true;
    }

    bool parse_star()
    {
        return tokenizer.parse_char('*');
    }

    bool parse_comma()
    {
        return tokenizer.parse_char(',');
    }

    void add_index(char c)
    {
        indices_list[index_head][indices_len[index_head]++] = repr(c);
    }

    void set_cc()
    {
        cc[index_head] = true;
    }

    int repr(char c)
    {
        char* seen_ptr;
        for (seen_ptr = seen; *seen_ptr != 0; seen_ptr++)
        {
            if (*seen_ptr == c)
            {
                return seen_ptr - seen;
            }
        }
        *seen_ptr++ = c;
        *seen_ptr = 0;
        return seen_ptr - seen - 1;
    }


    bool parse_indices()
    {
        bool success;
        char c;
        tokenizer.push();
        success = tokenizer.parse_alphabet(c);
        add_index(c);
        if (!success)
        {
            tokenizer.pop();
            expected_error("indices");
            return false;
        }
        tokenizer.stash();
        while (true)
        {
            tokenizer.push();
            success = parse_star();
            if (!success)
            {
                tokenizer.pop();
            }
            else
            {
                tokenizer.stash();
                set_cc();
            }
            tokenizer.push();
            success = tokenizer.parse_alphabet(c);
            if (!success)
            {
                tokenizer.pop();
                break;
            }
            add_index(c);
        }
        index_head++;
        return true;
    }

    private:

};


template <int N> struct multidim_looper
{
    public:
    int pos[N];
    int *lengths;
    int nthreads;
    int threadidx;

    __device__ multidim_looper(int *lengths,
                    const int& nthreads, 
                    const int& threadidx)
       : lengths(lengths), nthreads(nthreads), threadidx(threadidx)
    {
        pos[0] = threadidx - nthreads;
        for (int n=1; n<N; n++)
        {
            pos[n] = 0;
        }
    }

    void print()
    {
        printf("thread: %d/%d\n", threadidx, nthreads);
        for (int n=0;n<N; n++)
        {
            printf("element %d/%d ",pos[n]+1, lengths[n]);
        }
        printf("\n");
    }

    // return false, if one should stop here
    __device__ bool next()
    {
        int newidx = pos[0] + nthreads;
        int pos_inc;
        for (int n=0; n<N; n++)
        {
            pos[n] = newidx % lengths[n];
            pos_inc = newidx / lengths[n];
            if (n < N-1)
            {
                newidx = pos[n+1] + pos_inc;
            }
        }
        return (pos_inc == 0);
    }
};

template <int N, int nargs> struct strider
{
    int *strides_ai;
    const multidim_looper <N> &m;
    __device__ strider(const multidim_looper<N> &m, int *strides_ai) : m(m), strides_ai(strides_ai)
    {
    }

    __device__ int get_index(int arg) const
    {
        int ravel = 0;
        for (int n=0; n<N; n++)
        {
            ravel += m.pos[n] * strides_ai[arg * N + n];
        }
        return ravel;
    }
};

template <bool complex_out, bool add, size_t nind_out, size_t nind_in, size_t nargs> __global__ void multi_einsum_kernel_complex(int problems, int* size_out_pi, int* size_in_pi, int* strides_out_pai, int* strides_in_pai, int* cc, gpuDoubleComplex **arguments_pa)
{
    int problem_index = blockIdx.x;
    if (problem_index >= problems)
    {
        return;
    }
    int *size_out = size_out_pi + problem_index * nind_out;
    int *size_in = size_in_pi + problem_index * nind_in;
    int *strides_in = strides_in_pai + problem_index * nind_in * nargs;
    int *strides_out = strides_out_pai + problem_index * nind_out * nargs;
    multidim_looper<nind_out> out_index(size_out, blockDim.x, threadIdx.x);
    strider<nind_out, nargs> strider_out(out_index, strides_out);
    gpuDoubleComplex **arguments_a = arguments_pa + problem_index * nargs;                                                 
    while (out_index.next())
    {
        int out = strider_out.get_index((int)nargs - 1);
        multidim_looper<nind_in> in_index(size_in, 1, 0);
        strider<nind_in, nargs> strider_in(in_index, strides_in);
        gpuDoubleComplex sum = make_gpuDoubleComplex(0,0);
        while (in_index.next())
        {
            gpuDoubleComplex value = make_gpuDoubleComplex(1,0);
            for (int arg=0; arg < nargs -1; arg++)
            {
                gpuDoubleComplex arg_value = arguments_a[arg][strider_in.get_index(arg) + strider_out.get_index(arg)];
                if (cc[arg])
                    arg_value = gpuConj(arg_value);
                value = gpuCmul(value, arg_value);
            }
            sum = gpuCadd(sum, value);
        }

        if (add)
        {
            if (complex_out)
                arguments_a[nargs-1][out] = gpuCadd(arguments_a[nargs-1][out], sum);
            else
                ((double*)(arguments_pa[nargs-1 + problem_index * nargs]))[out] += sum.x;

        }
        else
        {
            if (complex_out)
                arguments_a[nargs-1][out] = sum;
            else
                 ((double*)(arguments_pa[nargs-1 + problem_index * nargs]))[out] = sum.x;
        }
    }
}

template <bool add, size_t nind_out, size_t nind_in, size_t nargs>
    __global__ void multi_einsum_kernel(int problems, 
                                        int* size_out_pi,
                                        int* size_in_pi,
                                        int* strides_out_pai,
                                        int* strides_in_pai,
                                        double** arguments_pa)
{
    int problem_index = blockIdx.x;
    if (problem_index >= problems)
    {
        return;
    }
    int *size_out = size_out_pi + problem_index * nind_out;
    int *size_in = size_in_pi + problem_index * nind_in;
    int *strides_in = strides_in_pai + problem_index * nind_in * nargs;
    int *strides_out = strides_out_pai + problem_index * nind_out * nargs;
    multidim_looper<nind_out> out_index(size_out, blockDim.x, threadIdx.x);
    strider<nind_out, nargs> strider_out(out_index, strides_out);
    double** arguments_a = arguments_pa + problem_index * nargs;                                                 
    while (out_index.next())
    {
        int out = strider_out.get_index((int)nargs - 1);
        multidim_looper<nind_in> in_index(size_in, 1, 0);
        strider<nind_in, nargs> strider_in(in_index, strides_in);
        double sum = 0;
        while (in_index.next())
        {
            double value = 1;
            for (int arg=0; arg < nargs -1; arg++)
            {
                value *= arguments_a[arg][strider_in.get_index(arg) + strider_out.get_index(arg)];
            }
            sum += value;
        }
        if (add)
        {
            arguments_a[nargs-1][out] += sum;
        }
        else
        {
            arguments_a[nargs-1][out] = sum;
        }
    }
}

template <typename T> struct dual_buffer
{
    T* cpu_ptr;
    T* gpu_ptr;
    dual_buffer(T* cpu_ptr, T* gpu_ptr)
        : cpu_ptr(cpu_ptr), gpu_ptr(gpu_ptr)
    {
    }
};

template <typename T> struct buffer
{
    int size;
    T* cpu_ptr;
    T* gpu_ptr;
    int allocated;
    int head;
     
    buffer(int size)
        : size(size), allocated(0), head(0)
    {
        gpuMalloc(&gpu_ptr, size * sizeof(T));
        cpu_ptr = (T*) malloc(size * sizeof(T));
    }

    dual_buffer<T> allocate(int n)
    {
        int current_head = head;
        dual_buffer<T> b(cpu_ptr + current_head,
                         gpu_ptr + current_head);
        head += n; 
        assert (head <= size);
        return b;
    }

    ~buffer()
    {
        gpuFree(gpu_ptr);
        free(cpu_ptr);
    }

    void copy_to_device()
    {
        gpuMemcpy(gpu_ptr, cpu_ptr, sizeof(T) * size, gpuMemcpyHostToDevice);
    }
};



typedef void(*multi_einsum_real_kernel_call)(int, int*, int* ,int*, int*, double**);
typedef void(*multi_einsum_complex_kernel_call)(int, int*, int* ,int*, int*, int*, gpuDoubleComplex**);

template<std::size_t N>
struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
void for_(F func, std::index_sequence<Is...>)
{
  using expander = int[];
  (void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
void for_(F func)
{
  for_(func, std::make_index_sequence<N>());
}


template <int max_out, int max_in, int max_args> struct kernel_funcs
{
    multi_einsum_real_kernel_call real_calls[2][max_out][max_in][max_args];
    multi_einsum_complex_kernel_call complex_calls[2][2][max_out][max_in][max_args];
    constexpr kernel_funcs(int nind_out, int nind_in, int arguments, bool add, bool is_complex_out)
        : nind_in(nind_in), nind_out(nind_out), arguments(arguments), add(add), is_complex_out(is_complex_out)
    {
        assert(nind_out < max_out);
        assert(nind_in < max_in);
        assert(arguments < max_args);
        for_<max_in>([&] (auto nind_in) 
            {
                for_<max_out>([&] (auto nind_out) 
                {
                    for_<max_args>([&] (auto args) 
                    {
                       real_calls[0][nind_out.value][nind_in.value][args.value] = &multi_einsum_kernel<false, nind_out.value, nind_in.value, args.value>;
                       real_calls[1][nind_out.value][nind_in.value][args.value] = &multi_einsum_kernel<true, nind_out.value, nind_in.value, args.value>;
                       complex_calls[0][0][nind_out.value][nind_in.value][args.value] = &multi_einsum_kernel_complex<false, false, nind_out.value, nind_in.value, args.value>;
                       complex_calls[0][1][nind_out.value][nind_in.value][args.value] = &multi_einsum_kernel_complex<false, true, nind_out.value, nind_in.value, args.value>;
                       complex_calls[1][0][nind_out.value][nind_in.value][args.value] = &multi_einsum_kernel_complex<true, false, nind_out.value, nind_in.value, args.value>;
                       complex_calls[1][1][nind_out.value][nind_in.value][args.value] = &multi_einsum_kernel_complex<true, true, nind_out.value, nind_in.value, args.value>;
                    });
                });
            });
    }

    int nind_in;
    int nind_out;
    int arguments;
    int add;
    int is_complex_out; 



    multi_einsum_real_kernel_call get_real_kernel()
    {
        return real_calls[add][nind_out][nind_in][arguments];
    }

    multi_einsum_complex_kernel_call get_complex_kernel()
    {
        return complex_calls[is_complex_out][add][nind_out][nind_in][arguments];
    }
};

extern "C"
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
                                char* error)
{
    buffer<int> intbuffer(2 * problems * arguments * maxind + arguments);
    buffer<double*> arguments_pa_buffer(problems * arguments);
    dual_buffer<double*> arguments_pa = arguments_pa_buffer.allocate(problems * arguments);
    
    //printf("%s\n", str);
    Tokenizer tokenizer(str);
    EinsumArgumentParser parser(tokenizer);
    parser.parse();
    parser.print_error();
    if (strlen(parser.error))
    {
        printf("error launch %s\n", parser.error);
        strcpy(error, parser.error); 
        return;
    }
    parser.print();

    int nind_out = parser.nindex_out();
    int nind_in = parser.nindex_in();
    //printf("number of arguments %d\n", arguments);
    //printf("iscomplex %d\n", is_complex);
    //printf("indices out:");
    //for (int n=0; n<nind_out; n++)
    //{
    //    printf(" %d", parser.index_out(n));
    //}
    //printf("\nindices in:");
    //for (int n=0; n<nind_in; n++)
    //{
    //    printf(" %d", parser.index_in(n));
    // }
    /*
    for (int p=0; p< problems; p++)
    {
        for (int a=0; a<arguments; a++)
        {
            for (int i=0; i<maxind; i++)
            {
                printf("p=%d a=%d i=%d size=%d stride=%d\n", p,a,i,
                       dimensions_pai[p * (arguments * maxind) + 4 * a + i],
                       strides_pai[p * (arguments * maxind) + 4 * a + i]);
            }
        }
    }*/

    //printf("\n");
    
    dual_buffer<int> size_in_pi = intbuffer.allocate(problems * nind_in);
    dual_buffer<int> size_out_pi = intbuffer.allocate(problems * nind_out);
    dual_buffer<int> strides_in_pai = intbuffer.allocate(problems * arguments * nind_in);
    dual_buffer<int> strides_out_pai = intbuffer.allocate(problems * arguments * nind_out);
    dual_buffer<int> cc_a = intbuffer.allocate(arguments);
    for (int a=0; a<arguments; a++)
    {
        cc_a.cpu_ptr[a] = parser.cc[a];
    } 
    for (int p = 0; p < problems; p++)
    {
        for (int a=0; a<arguments; a++)
        {
            arguments_pa.cpu_ptr[p*arguments + a] = array_pointers_pa[p*arguments + a];
        }
        for (int i=0; i < nind_out; i++)
        {
            int index_out_size = dimensions_pai[ p * (arguments * maxind) + maxind * (arguments-1) + i]; 
            if (index_out_size == 0)
            {
                char error_str[256];
                sprintf(error_str, "Too few indices in output of einsum for problem %d (indexing starts at 0).", p);
                strcpy(error, error_str);
                return;
            }
            size_out_pi.cpu_ptr[p * nind_out + i ] = index_out_size;
            //printf("p: %d outind: %d size: %d\n", p, i, index_out_size);
            for (int a=0; a < arguments; a++)
            {
               int stride = 0;
               for (int locind=0; locind < parser.indices_len[a]; locind++)
               {
                   if (parser.indices_list[a][locind] == parser.indices_list[arguments-1][i])
                   {
                       stride += strides_pai[ p * (arguments * maxind) + maxind * a + locind];
                   }
               }
               //printf("strides p %d a %d i %d: %d\n", p, a, i, stride);
               strides_out_pai.cpu_ptr[p * (nind_out*arguments) + a * nind_out + i ] = stride; 

            }
        }
        for (int i=0; i < nind_in; i++)
        {
            for (int a=0; a < arguments; a++)
            {
               int stride = 0;
               for (int locind=0; locind < parser.indices_len[a]; locind++)
               {
                   if (parser.indices_list[a][locind] == parser.index_in(i))
                   {
                       stride += strides_pai[ p * (arguments * maxind) + maxind * a + locind];
                       int index_in_size = dimensions_pai[ p * (arguments * maxind) + maxind * a + locind]; // move inside stride
                       if (index_in_size == 0)
                       {
                           char error_str[256];
                           if (a < arguments -1)
                           {
                               sprintf(error_str, "Too few indices in argument %d of einsum.", a+1);
                           }
                           else
                           {
                               sprintf(error_str, "Too few indices in output of einsum.");
                           }
                           strcpy(error, error_str);
                       }
                       size_in_pi.cpu_ptr[p * nind_in + i ] = index_in_size;
                   }
               }
               //printf("strides p %d a %d i %d: %d\n", p, a, i, stride);
               strides_in_pai.cpu_ptr[p * (nind_in*arguments) + a * nind_in + i ] = stride;

            //printf("p: %d in_ind: %d size: %d\n", p, i, index_in_size);

            }
        }

        /*
        printf("Problem %d\nOuter size:",p);
        for (int i=0;i<nind_out; i++)
        {
            printf(" %d", size_out_pi.cpu_ptr[p * nind_out + i]);
        }
        printf("\nInner size:");
        for (int i=0;i<nind_in; i++)
        {
            printf(" %d", size_in_pi.cpu_ptr[p * nind_in + i]);
        }

        printf("\nStrides for all arguments:\n");

        for (int a=0; a<arguments; a++)
        {
            printf("out:");
            for (int i=0;i<nind_out;i++)
            {
                printf(" %d", strides_out_pai.cpu_ptr[p * (nind_out*arguments) + a * nind_out + i]);
            }
            printf("in:");
            for (int i=0;i<nind_in;i++)
            {
                printf(" %d", strides_in_pai.cpu_ptr[p * (nind_in*arguments) + a * nind_in + i]);
            }
            printf("\n");
        }
        */
    }

    intbuffer.copy_to_device();
    arguments_pa_buffer.copy_to_device();
 
    if (nind_out >= 4)
    {
        strcpy(error, "Too many output indices. Edit template parameters of kernel_funcs to allow more.");
        return;
    } 
    if (nind_in >= 4)
    {
        strcpy(error, "Too many inner indices. Edit template parameters of kernel_funcs.");
        return;
    } 
    if (nind_in >= 5)
    {
        strcpy(error, "Too many arguments. Edit template parameters of kernel_funcs.");
        return;
    } 
    kernel_funcs <4, 4, 5> f(nind_out, nind_in, arguments, add, is_complex_out);

    if (is_complex)
    {
        gpuLaunchKernel(f.get_complex_kernel(),
                        dim3(problems),
                        dim3(256),
                        0, 0,
                        problems,
                        size_out_pi.gpu_ptr,
                        size_in_pi.gpu_ptr,
                        strides_out_pai.gpu_ptr,
                        strides_in_pai.gpu_ptr,
                        cc_a.gpu_ptr,
                        (gpuDoubleComplex**) arguments_pa.gpu_ptr);
    }
    else
    {
        gpuLaunchKernel(f.get_real_kernel(),
                        dim3(problems),
                        dim3(256),
                        0, 0,
                        problems,
                        size_out_pi.gpu_ptr,
                        size_in_pi.gpu_ptr,
                        strides_out_pai.gpu_ptr,
                        strides_in_pai.gpu_ptr,
                        arguments_pa.gpu_ptr);
    }
}

extern "C"
PyObject* multi_einsum_gpu(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static const char *kwlist[] = {"string", "arguments", "out", "add", NULL};
    char* string;
    char error[256] = "";


    // arguments is expected to be list of tuples
    PyObject* arguments_obj;
    PyObject* out_obj = NULL;
    PyObject* add_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|OO", (char**) kwlist, 
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
            double* array_ptr = (double*) Array_DATA(cupy_array);
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
                               error);
    free(array_pointers);
    free(dimensions);
    if (strlen(error))
    {
        printf("Error %s\n", error);
        PyErr_SetString(PyExc_RuntimeError, error);
        return NULL;
    }
    Py_RETURN_NONE;
error:
    free(array_pointers);
    free(dimensions);
    return NULL;    
}


