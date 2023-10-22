#include "../gpu.h"
#include "../gpu-complex.h"
#include "numpy/arrayobject.h"
#include "assert.h"
#include <cstdlib>

#define BETA   0.066725
#define GAMMA  0.031091
#define MU     0.2195164512208958 // PBE mod in libxc
//#define MU     0.2195149727645171 from libxc
#define C2     0.26053088059892404
#define C0I    0.238732414637843
#define C1    -0.45816529328314287
#define CC1    1.9236610509315362
#define CC2    2.5648814012420482
#define IF2    0.58482236226346462
#define C3     0.10231023756535741
#define C0     4.1887902047863905
#define THIRD  0.33333333333333333
#define NMIN   1.0E-10

class Parser
{
    public:
        Parser(const char* str) : str(str), chr(str), error(""), head(0), error_set(0)
        {
        }

        void print_error()
        {
            printf("%s\n", error);
        }

    protected:
        char peek()
        {
            return *chr;
        }

        char next()
        {
            char c = *chr;
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

        void expected_error(const char* expected_str)
        {
            if (error_set)
            {
                printf("Internal error. Error already set:");
                print_error();
            }
            error_set = true;
            char c = next();
            if (c != 0)
            {
                sprintf(error, "Expected %s instead of '%c'.", expected_str, c);
            }
            else
            {
                sprintf(error, "Expected %s instead of end of string.", expected_str);
            }
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

    private:
        const char* str;
        const char* chr;
        char error[50];
        bool error_set;

        const char* stack[10];
        int head;
};

class EinsumArgumentParser : public Parser
{
    public:
    int indices_list[10][10];
    int indices_len[10];
    char seen[10];
    int index_head;

    EinsumArgumentParser(const char* str) 
        : Parser(str), index_head(0), seen("")
    {
        for (int i=0;i<10;i++)
            indices_len[i] = 0;
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
        push();
        success = parse_char('-');
        if (!success)
        {
            pop();
            expected_error("-");
            return false;
        }
        stash();
        push();
        success = parse_char('>');
        if (!success)
        {
            pop();
            expected_error("> after -");
            return false;
        }
        stash();
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
            push();
            success = parse_comma();
            if (!success)
            {
                pop();
                break;
            }
            stash();
            push();
            success = parse_indices();
            if (!success)
            {
                pop();
                expected_error("indices");
                return false;
            }
            stash();
        }
        return true;
    }

    bool parse_comma()
    {
        return parse_char(',');
    }

    void add_index(char c)
    {
        indices_list[index_head][indices_len[index_head]++] = repr(c);
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
        push();
        success = parse_alphabet(c);
        add_index(c);
        if (!success)
        {
            pop();
            expected_error("indices");
            return false;
        }
        stash();
        while (true)
        {
            push();
            success = parse_alphabet(c);
            if (!success)
            {
                pop();
                break;
            }
            add_index(c);
        }
        index_head++;
        return true;
    }

    private:

};

/*
template <int max_indices> __global__ 
    void multi_einsum_kernel_3index_4arg(int problems, int na, int nb, int nc,
                                                       int* strides_pac)
{
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g < problems)
    {
        double* array_pointer_table = array_pointers[g * 4]; // (shape 4x problems)
        double* A = array_pointer_table[0];
        double* B = array_pointer_table[1];
        double* C = array_pointer_table[2];
        double* D = array_pointer_table[3];
        double* strides_ac = strides_pac + g * (3 * 3);
        
        for (int a=0; a<na; a++) 
        for (int b=0; b<nb; b++) 
        for (int c=0; c<nc; c++)
        {
            int indexA = a * strides_ac[0] + b * strides_ac[1] + c * strides_ac[2];
            int indexB = a * strides_ac[3] + b * strides_ac[4] + c * strides_ac[5];
            int indexC = a * strides_ac[6] + b * strides_ac[7] + c * strides_ac[8];
            int indexD = a * strides_ac[9] + b * strides_ac[10] + c * strides_ac[11];
            D[indexD] += A[indexA] * B[indexB] * C[indexC]; 
        } 
    }
}
*/

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

template <bool add, int nind_out, int nind_in, int nargs> __global__ void multi_einsum_kernel(int problems, int* size_out_pi, int* size_in_pi, int* strides_out_pai, int* strides_in_pai, double** arguments_pa)
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
        int out = strider_out.get_index(nargs - 1); // Out is the final argument
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
        arguments_a[nargs-1][out] += sum;
        else
        arguments_a[nargs-1][out] = sum;

        //printf("Storing %f to %d.\n", sum, out);
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
        dual_buffer b(cpu_ptr + current_head,
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


constexpr void(*multieinsum_214)(bool, int, int*, int*, int*, int*, double**) = &multi_einsum_kernel<false, 2, 1, 4>;
constexpr void(*multieinadd_214)(bool, int, int*, int*, int*, int*, double**) = &multi_einsum_kernel<true, 2, 1, 4>;

extern "C"
void multi_einsum_launch_kernel(char* str,
                                int problems,  // p index
                                int arguments, // a index
                                int maxind,    // i index
                                int* dimensions_pai,
                                int* strides_pai,
                                double** array_pointers_pa,
                                int add)
{
    buffer<int> intbuffer(2 * problems * arguments * maxind);
    buffer<double*> arguments_pa_buffer(problems * arguments);
    dual_buffer<double*> arguments_pa = arguments_pa_buffer.allocate(problems * arguments);
    
    printf("%s\n", str);
    EinsumArgumentParser parser(str);
    parser.parse();
    parser.print_error();
    parser.print();

    int nind_out = parser.nindex_out();
    int nind_in = parser.nindex_in();
    printf("number of arguments %d ", arguments);
    printf("indices out:");
    for (int n=0; n<nind_out; n++)
    {
        printf(" %d", parser.index_out(n));
    }
    printf("\nindices in:");
    for (int n=0; n<nind_in; n++)
    {
        printf(" %d", parser.index_in(n));
    }

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
    }

    printf("\n");
    
    dual_buffer<int> size_in_pi = intbuffer.allocate(problems * nind_in);
    dual_buffer<int> size_out_pi = intbuffer.allocate(problems * nind_out);
    dual_buffer<int> strides_in_pai = intbuffer.allocate(problems * arguments * nind_in);
    dual_buffer<int> strides_out_pai = intbuffer.allocate(problems * arguments * nind_out);
    
    for (int p = 0; p < problems; p++)
    {
        for (int a=0; a<arguments; a++)
        {
            arguments_pa.cpu_ptr[p*arguments + a] = array_pointers_pa[p*arguments + a];
        }
        for (int i=0; i < nind_out; i++)
        {
            int index_out_size = dimensions_pai[ p * (arguments * maxind) + maxind * (arguments-1) + i]; 
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
                       size_in_pi.cpu_ptr[p * nind_in + i ] = index_in_size;
                   }
               }
               //printf("strides p %d a %d i %d: %d\n", p, a, i, stride);
               strides_in_pai.cpu_ptr[p * (nind_in*arguments) + a * nind_in + i ] = stride;

            //printf("p: %d in_ind: %d size: %d\n", p, i, index_in_size);

            }
        }

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

        printf("Strides for all arguments:\n");

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

        /*
        for (int i=0; i < nind_in; i++)
        {
            int size = parser.index_in_size(i);
            size_in_pi.cpu_ptr[p * maxind + ind ] = dimensions_pai[ p * (arguments * maxind) + maxind * (arguments-1) + ind]; 
            //strides_out_pi.cpu_ptr[p * maxind + i ] = strides_pai[ p * (arguments * maxind) + maxind * (arguments-1) + i];
            printf("p: %d inind: %d size: %d\n", p, i, size_in_pi.cpu_ptr[p*maxind+i]);
        }
        */
    }

    intbuffer.copy_to_device();
    arguments_pa_buffer.copy_to_device();
  
    if ((nind_out == 2) && (nind_in == 1) && (arguments == 4))
    {
        if (add) 
        gpuLaunchKernel(multieinadd_214,
                        dim3(problems),
                        dim3(256),
                        0, 0,
                        problems,
                        size_out_pi.gpu_ptr,
                        size_in_pi.gpu_ptr,
                        strides_out_pai.gpu_ptr,
                        strides_in_pai.gpu_ptr,
                        arguments_pa.gpu_ptr);
        else
        gpuLaunchKernel(multieinsum_214,
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
    else
    {
        printf("Einsum not implemented for nind_out %d nind_in %d arguments %d.\n", nind_out, nind_in, arguments);
    }
}

template <bool gga> __device__ double pbe_exchange(double n, double rs, double a2,
                                                   double* dedrs, double* deda2)
{
    double e = C1 / rs;
    *dedrs = -e / rs;
    if (gga)
    {
        double kappa = 0.804;
        double c = C2 * rs / n;
        c *= c;
        double s2 = a2 * c;
        double x = 1.0 + MU * s2 / kappa;
        double Fx = 1.0 + kappa - kappa / x;
        double dFxds2 = MU / (x * x);
        double ds2drs = 8.0 * c * a2 / rs;
        *dedrs = *dedrs * Fx + e * dFxds2 * ds2drs;
        *deda2 = e * dFxds2 * c;
        e *= Fx;
    }
    return e;
}


__device__ double G(double rtrs, double A, double alpha1,
                    double beta1, double beta2, double beta3, double beta4,
                    double* dGdrs)
{
  double Q0 = -2.0 * A * (1.0 + alpha1 * rtrs * rtrs);
  double Q1 = 2.0 * A * rtrs * (beta1 + 
                                rtrs * (beta2 + 
                                        rtrs * (beta3 + 
                                                rtrs * beta4)));
  double G1 = Q0 * log(1.0 + 1.0 / Q1);
  double dQ1drs = A * (beta1 / rtrs + 2.0 * beta2 +
                       rtrs * (3.0 * beta3 + 4.0 * beta4 * rtrs));
  *dGdrs = -2.0 * A * alpha1 * G1 / Q0 - Q0 * dQ1drs / (Q1 * (Q1 + 1.0));
  return G1;
}


template <bool gga, int nspin> __device__ double pbe_correlation(double n, double rs, double zeta, double a2, 
		                                       double* dedrs, double* dedzeta, double* deda2)
{
  bool spinpol = nspin == 2;
  double rtrs = sqrt(rs);
  double de0drs;
  double e0 = G(rtrs, GAMMA, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294,
		&de0drs);
  double e;
  double xp = 117.0;
  double xm = 117.0;
  if (spinpol)
    {
      double de1drs;
      double e1 = G(rtrs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662,
                    0.62517, &de1drs);
      double dalphadrs;
      double alpha = -G(rtrs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026,
                        0.49671, &dalphadrs);
      dalphadrs = -dalphadrs;
      double zp = 1.0 + zeta;
      double zm = 1.0 - zeta;
      xp = pow(zp, THIRD);
      xm = pow(zm, THIRD);
      double f = CC1 * (zp * xp + zm * xm - 2.0);
      double f1 = CC2 * (xp - xm);
      double zeta2 = zeta * zeta;
      double zeta3 = zeta2 * zeta;
      double zeta4 = zeta2 * zeta2;
      double x = 1.0 - zeta4;
      *dedrs = (de0drs * (1.0 - f * zeta4) + 
               de1drs * f * zeta4 +
               dalphadrs * f * x * IF2);
      *dedzeta = (4.0 * zeta3 * f * (e1 - e0 - alpha * IF2) +
                 f1 * (zeta4 * e1 - zeta4 * e0 + x * alpha * IF2));
      e = e0 + alpha * IF2 * f * x + (e1 - e0) * f * zeta4;
    }
  else
    {
      *dedrs = de0drs;
      e = e0;
    }
  if (gga)
    {
      double n2 = n * n;
      double t2;
      double y;
      double phi = 117.0;
      double phi2 = 117.0;
      double phi3 = 117.0;
      if (spinpol)
        {
          phi = 0.5 * (xp * xp + xm * xm);
          phi2 = phi * phi;
          phi3 = phi * phi2;
          t2 = C3 * a2 * rs / (n2 * phi2);
          y = -e / (GAMMA * phi3);
        }
      else
        {
          t2 = C3 * a2 * rs / n2;
          y = -e / GAMMA;
        }
      double x = exp(y);
      double A;
      if (x != 1.0)
        A = BETA / (GAMMA * (x - 1.0)); 
      else
        A = BETA / (GAMMA * y);
      double At2 = A * t2;
      double nom = 1.0 + At2;
      double denom = nom + At2 * At2;
      double H = GAMMA * log( 1.0 + BETA * t2 * nom / (denom * GAMMA));
      double tmp = (GAMMA * BETA /
                    (denom * (BETA * t2 * nom + GAMMA * denom)));
      double tmp2 = A * A * x / BETA;
      double dAdrs = tmp2 * *dedrs;
      if (spinpol)
        {
          H *= phi3;
          tmp *= phi3;
          dAdrs /= phi3;
        }
      double dHdt2 = (1.0 + 2.0 * At2) * tmp;
      double dHdA = -At2 * t2 * t2 * (2.0 + At2) * tmp;
      *dedrs += dHdt2 * 7 * t2 / rs + dHdA * dAdrs;
      *deda2 = dHdt2 * C3 * rs / n2;
      if (spinpol)
        {
          double dphidzeta = (1.0 / xp - 1.0 / xm) / 3.0;
          double dAdzeta = tmp2 * (*dedzeta - 
				   3.0 * e * dphidzeta / phi) / phi3;
          *dedzeta += ((3.0 * H / phi - dHdt2 * 2.0 * t2 / phi ) * dphidzeta +
                      dHdA * dAdzeta);
          *deda2 /= phi2;
        }
      e += H;
    }
  return e;
}


template <int nspin, bool gga> __global__ void evaluate_ldaorgga_kernel(int ng,
                                                                        double* n_sg,
                                                                        double* v_sg,
                                                                        double* e_g,
                                                                        double* sigma_xg,
                                                                        double* dedsigma_xg)
{
    int g = threadIdx.x + blockIdx.x * blockDim.x;
    if (g >= ng)
    {
        return;
    }
    if (nspin == 1)
    {
        double n = n_sg[g];
        if (n < NMIN)
          n = NMIN;
        double rs = pow(C0I / n, THIRD);
        double dexdrs;
        double dexda2;
        double ex;
        double decdrs;
        double decda2;
        double ec;
        if (gga)
          {
            double a2 = sigma_xg[g];
            ex = pbe_exchange<gga>(n, rs, a2, &dexdrs, &dexda2);
            ec = pbe_correlation<gga, nspin>(n, rs, 0.0, a2, &decdrs, 0, &decda2);
            dedsigma_xg[g] = n * (dexda2 + decda2);
          }
        else
          {
            ex = pbe_exchange<gga>(n, rs, 0.0, &dexdrs, 0);
            ec = pbe_correlation<gga, nspin>(n, rs, 0.0, 0.0, &decdrs, 0, 0);
          }
        e_g[g] = n * (ex + ec);
        v_sg[g] += ex + ec - rs * (dexdrs + decdrs) / 3.0;
    }
    else
    {
        const double* na_g = n_sg;
        double* va_g = v_sg;
        const double* nb_g = na_g + ng;
        double* vb_g = va_g + ng;

        const double* sigma0_g = 0;
        const double* sigma1_g = 0;
        const double* sigma2_g = 0;
        double* dedsigma0_g = 0;
        double* dedsigma1_g = 0;
        double* dedsigma2_g = 0;

        if (gga)
        {
            sigma0_g = sigma_xg;
            sigma1_g = sigma0_g + ng;
            sigma2_g = sigma1_g + ng;
            dedsigma0_g = dedsigma_xg;
            dedsigma1_g = dedsigma0_g + ng;
            dedsigma2_g = dedsigma1_g + ng;
        }

        double na = 2.0 * na_g[g];
        if (na < NMIN)
          na = NMIN;
        double rsa = pow(C0I / na, THIRD);
        double nb = 2.0 * nb_g[g];
        if (nb < NMIN)
          nb = NMIN;
        double rsb = pow(C0I / nb, THIRD);
        double n = 0.5 * (na + nb);
        double rs = pow(C0I / n, THIRD);
        double zeta = 0.5 * (na - nb) / n;
        double dexadrs;
        double dexada2;
        double exa;
        double dexbdrs;
        double dexbda2;
        double exb;
        double decdrs;
        double decdzeta;
        double decda2;
        double ec;
        if (gga)
        {
            exa = pbe_exchange<gga>(na, rsa, 4.0 * sigma0_g[g],
                               &dexadrs, &dexada2);
            exb = pbe_exchange<gga>(nb, rsb, 4.0 * sigma2_g[g],
                                   &dexbdrs, &dexbda2);
            double a2 = sigma0_g[g] + 2 * sigma1_g[g] + sigma2_g[g];
            ec = pbe_correlation<gga, nspin>(n, rs, zeta, a2, 
                                             &decdrs, &decdzeta, &decda2);
            dedsigma0_g[g] = 2 * na * dexada2 + n * decda2;
            dedsigma1_g[g] = 2 * n * decda2;
            dedsigma2_g[g] = 2 * nb * dexbda2 + n * decda2;
        }
        else
        {
           exa = pbe_exchange<gga>(na, rsa, 0.0, &dexadrs, 0);
           exb = pbe_exchange<gga>(nb, rsb, 0.0, &dexbdrs, 0);
           ec = pbe_correlation<gga, nspin>(n, rs, zeta, 0.0, 
                                            &decdrs, &decdzeta, 0);
        }

        e_g[g] = 0.5 * (na * exa + nb * exb) + n * ec;
        va_g[g] += (exa + ec -
                    (rsa * dexadrs + rs * decdrs) / 3.0 -
                    (zeta - 1.0) * decdzeta);
        vb_g[g] += (exb + ec -
                    (rsb * dexbdrs + rs * decdrs) / 3.0 -
                    (zeta + 1.0) * decdzeta);
    }
}

// The define wrappers do not allow special characters for the first argument
// Hence, here defining an expression in such way, that the first argument can be
// a well defined identifier, and the preprocessor macro parses it correctly.
constexpr void(*LDA_SPINPAIRED)(int, double*, double*, double*, double*, double*) = &evaluate_ldaorgga_kernel<1, false>;
constexpr void(*LDA_SPINPOLARIZED)(int, double*, double*, double*, double*, double*) = &evaluate_ldaorgga_kernel<2, false>;
constexpr void(*PBE_SPINPAIRED)(int, double*, double*, double*, double*, double*) = &evaluate_ldaorgga_kernel<1, true>;
constexpr void(*PBE_SPINPOLARIZED)(int, double*, double*, double*, double*, double*) = &evaluate_ldaorgga_kernel<2, true>;

extern "C"
void evaluate_pbe_launch_kernel(int nspin, int ng,
                                double* n,
                                double* v,
                                double* e,
                                double* sigma,
                                double* dedsigma)
{
    if (nspin == 1)
    {
        gpuLaunchKernel(PBE_SPINPAIRED,
                        dim3((ng+255)/256),
                        dim3(256),
                        0, 0,
                        ng,
                        n, v, e, sigma, dedsigma);
    }
    else if (nspin == 2) 
    {
        gpuLaunchKernel(PBE_SPINPOLARIZED,
                        dim3((ng+255)/256),
                        dim3(256),
                        0, 0,
                        ng,
                        n, v, e, sigma, dedsigma);
    }
}

extern "C"
void evaluate_lda_launch_kernel(int nspin, int ng,
                                double* n,
                                double* v,
                                double* e)
{
    if (nspin == 1)
    {
        gpuLaunchKernel(LDA_SPINPAIRED,
                        dim3((ng+255)/256),
                        dim3(256),
                        0, 0,
                        ng,
                        n, v, e, NULL, NULL);
    }
    else if (nspin == 2) 
    {
        gpuLaunchKernel(LDA_SPINPOLARIZED,
                        dim3((ng+255)/256),
                        dim3(256),
                        0, 0,
                        ng,
                        n, v, e, NULL, NULL);
    }
}

__global__ void pw_insert_many_16(int nb,
                                  int nG,
                                  int nQ,
                                  gpuDoubleComplex* c_nG,
                                  npy_int32* Q_G,
                                  double scale,
                                  gpuDoubleComplex* tmp_nQ)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int b = threadIdx.y + blockIdx.y * blockDim.y;
    __shared__ npy_int32 locQ_G[16];
    if (threadIdx.y == 0)
        locQ_G[threadIdx.x] = Q_G[G];
    __syncthreads();

    if ((G < nG) && (b < nb))
    {
        npy_int32 Q = locQ_G[threadIdx.x];
        tmp_nQ[Q + b * nQ] = gpuCmulD(c_nG[G + b * nG], scale);
    }
}

__global__ void add_to_density_16(int nb,
                                  int nR,
                                  double* f_n,
                                  gpuDoubleComplex* psit_nR,
                                  double* rho_R)
{
    //int b = threadIdx.x + blockIdx.x * blockDim.x;
    int R = threadIdx.x + blockIdx.x * blockDim.x;
    if (R < nR)
    {
        double rho = 0.0;
        for (int b=0; b< nb; b++)
        {
            int idx = b * nR + R;
            rho += f_n[b] * (psit_nR[idx].x * psit_nR[idx].x + psit_nR[idx].y * psit_nR[idx].y);
        }
        rho_R[R] += rho;
    }
}


__global__ void pw_insert_16(int nG,
                             int nQ,
                             gpuDoubleComplex* c_G,
                             npy_int32* Q_G,
                             double scale,
                             gpuDoubleComplex* tmp_Q)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    if (G < nG)
        tmp_Q[Q_G[G]] = gpuCmulD(c_G[G], scale);
}

extern "C" void gpawDeviceSynchronize()
{
    gpuDeviceSynchronize();
}

extern "C"
void add_to_density_gpu_launch_kernel(int nb,
                                      int nR,
                                      double* f_n,
                                      gpuDoubleComplex* psit_nR,
                                      double* rho_R)
{
    gpuLaunchKernel(add_to_density_16,
                    dim3((nR+255)/256),
                    dim3(256),
                    0, 0,
                    nb, nR,
                    f_n,
                    psit_nR,
                    rho_R);
}

extern "C"
void pw_insert_gpu_launch_kernel(
                             int nb,
                             int nG,
                             int nQ,
                             double* c_nG,
                             npy_int32* Q_G,
                             double scale,
                             double* tmp_nQ)
{
    if (nb == 1)
    {
       gpuLaunchKernel(pw_insert_16,
                       dim3((nG+15)/16, (nb+15)/16),
                       dim3(16, 16),
                       0, 0,
                       nG, nQ,
                       (gpuDoubleComplex*) c_nG, Q_G,
                       scale,
                       (gpuDoubleComplex*) tmp_nQ);
    }
    else
    {
       gpuLaunchKernel(pw_insert_many_16,
                       dim3((nG+15)/16, (nb+15)/16),
                       dim3(16, 16),
                       0, 0,
                       nb, nG, nQ,
                       (gpuDoubleComplex*) c_nG,
                       Q_G,
                       scale,
                       (gpuDoubleComplex*) tmp_nQ);
    }
}


__global__ void pwlfc_expand_kernel_8(double* f_Gs,
                                       gpuDoubleComplex *emiGR_Ga,
                                       double *Y_GL,
                                       int* l_s,
                                       int* a_J,
                                       int* s_J,
                                       int* I_J,
                                       double* f_GI,
                                       int nG,
                                       int nJ,
                                       int nL,
                                       int nI,
                                       int natoms,
                                       int nsplines,
                                       bool cc)
{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int J = threadIdx.y + blockIdx.y * blockDim.y;
    gpuDoubleComplex imag_powers[4] = {make_gpuDoubleComplex(1.0,0),
                                       make_gpuDoubleComplex(0.0,-1.0),
                                       make_gpuDoubleComplex(-1.0,0),
                                       make_gpuDoubleComplex(0, 1.0)};
    if ((G < nG) && (J < nJ))
    {
        f_Gs += G*nsplines;
        emiGR_Ga += G*natoms;
        Y_GL += G*nL;
        f_GI += G*nI*2 + I_J[J];

        int s = s_J[J];
        int l = l_s[s];
        gpuDoubleComplex f1 = gpuCmulD(gpuCmul(emiGR_Ga[a_J[J]],
                                               imag_powers[l % 4]),
                                       f_Gs[s]);
        for (int m = 0; m < 2 * l + 1; m++) {
            gpuDoubleComplex f = gpuCmulD(f1, Y_GL[l * l + m]);
            f_GI[0] = f.x;
            f_GI[nI] = cc ? -f.y : f.y;
            f_GI++;
        }
    }
}

__global__ void pwlfc_expand_kernel_16(double* f_Gs,
                                       gpuDoubleComplex *emiGR_Ga,
                                       double *Y_GL,
                                       int* l_s,
                                       int* a_J,
                                       int* s_J,
                                       int* I_J,
                                       double* f_GI,
                                       int nG,
                                       int nJ,
                                       int nL,
                                       int nI,
                                       int natoms,
                                       int nsplines,
                                       bool cc)

{
    int G = threadIdx.x + blockIdx.x * blockDim.x;
    int J = threadIdx.y + blockIdx.y * blockDim.y;
    gpuDoubleComplex imag_powers[4] = {make_gpuDoubleComplex(1.0,0),
                                       make_gpuDoubleComplex(0.0,-1.0),
                                       make_gpuDoubleComplex(-1.0,0),
                                       make_gpuDoubleComplex(0, 1.0)};
    if ((G < nG) && (J < nJ))
    {
        f_Gs += G*nsplines;
        emiGR_Ga += G*natoms;
        Y_GL += G*nL;
        f_GI += (G*nI + I_J[J])*2;
        int s = s_J[J];
        int l = l_s[s];
        gpuDoubleComplex f1 = gpuCmulD(gpuCmul(emiGR_Ga[a_J[J]],
                                               imag_powers[l % 4]),
                                       f_Gs[s]);
        for (int m = 0; m < 2 * l + 1; m++) {
            gpuDoubleComplex f = gpuCmulD(f1, Y_GL[l * l + m]);
            *f_GI++ = f.x;
            *f_GI++ = cc ? -f.y : f.y;
        }
    }
}

// outP_ani[a] = \sum_A H_aii[a] P_ani[a]
__global__ void dH_aii_times_P_ani_16(int nA, int nn, int nI, 
                                      npy_int32* ni_a, double* dH_aii_dev, 
                                      gpuDoubleComplex* P_ani_dev,
                                      gpuDoubleComplex* outP_ani_dev)
{
    int n1 = threadIdx.x + blockIdx.x * blockDim.x;
    if (n1 < nn) {
        double* dH_ii = dH_aii_dev;
        int I = 0;        
        for (int a=0; a< nA; a++)
        {
            int ni = ni_a[a];
            int Istart = I;
            for (int i=0; i< ni; i++)
            {
                gpuDoubleComplex* outP_ni = outP_ani_dev + n1 * nI + I;
                gpuDoubleComplex result = make_gpuDoubleComplex(0.0, 0.0);
                gpuDoubleComplex* P_ni = P_ani_dev + n1 * nI + Istart;
                for (int i2=0; i2 < ni; i2++)
                {
                   //printf("%d %d %d %d %g\n", n1, a, i, i2, dH_ii[i2 * ni + i]);
                   gpuDoubleComplex item = gpuCmulD(*P_ni, dH_ii[i2 * ni + i]);
                   result.x += item.x;
                   result.y += item.y;
                   P_ni++;
                }
                outP_ni->x = result.x;
                outP_ni->y = result.y;
                I++;
            }
            //P_ni += ni;
            //outP_ni += ni;
            dH_ii += ni * ni;
        }
    }
}



extern "C"
void dH_aii_times_P_ani_launch_kernel(int nA, int nn,
                                      int nI, npy_int32* ni_a, 
                                      double* dH_aii_dev, 
                                      gpuDoubleComplex* P_ani_dev,
                                      gpuDoubleComplex* outP_ani_dev)
{
    gpuLaunchKernel(dH_aii_times_P_ani_16,
                    dim3((nn+255)/256),
                    dim3(256),
                    0, 0,
                    nA, nn, nI, ni_a, dH_aii_dev,
                    P_ani_dev, outP_ani_dev);
}



extern "C"
void pwlfc_expand_gpu_launch_kernel(int itemsize,
                                    double* f_Gs,
                                    gpuDoubleComplex *emiGR_Ga,
                                    double *Y_GL,
                                    int* l_s,
                                    int* a_J,
                                    int* s_J,
                                    double* f_GI,
                                    int* I_J,
                                    int nG,
                                    int nJ,
                                    int nL,
                                    int nI,
                                    int natoms,
                                    int nsplines,
                                    bool cc)
{
    if (itemsize == 16)
    {
        gpuLaunchKernel(pwlfc_expand_kernel_16,
                        dim3((nG+15)/16, (nJ+15)/16),
                        dim3(16, 16),
                        0, 0,
                        f_Gs,
                        emiGR_Ga,
                        Y_GL,
                        l_s,
                        a_J,
                        s_J,
                        I_J,
                        f_GI,
                        nG,
                        nJ,
                        nL,
                        nI,
                        natoms,
                        nsplines,
                        cc);
    }
    else
    {
        gpuLaunchKernel(pwlfc_expand_kernel_8,
                        dim3((nG+15)/16, (nJ+15)/16),
                        dim3(16, 16),
                        0, 0,
                        f_Gs,
                        emiGR_Ga,
                        Y_GL,
                        l_s,
                        a_J,
                        s_J,
                        I_J,
                        f_GI,
                        nG,
                        nJ,
                        nL,
                        nI,
                        natoms,
                        nsplines,
                        cc);
    }
    //gpuDeviceSynchronize();
}
