include ../make.inc

CUSRC = 

CSRC  = kernel_test.c
PROG  = kernel_test


all: libgpawcuda $(PROG)

ALLSRC = $(CSRC) $(CUSRC)

ALLOBJ = $(CSRC:.c=.o) $(CUSRC:.cu=.o)

NVLDFLAGS=-L.. -lgpaw-cuda -L$(PYTHONLIBdir) -lpython2.6 -lutil -L$(CUDALIBdir) -lcuda -lcublas -lmkl_intel_lp64 -lgfortran -lmkl_sequential -lmkl_core

-include make.dep

.PHONY: depend clean libgpawcuda


libgpawcuda: 
	@cd .. && $(MAKE)


$(PROG): $(ALLSRC) ../$(CUGPAWLIB)
	$(NVCC) $(NVCCFLAGS) $(NVLDFLAGS) $(ALLSRC) -o $@

clean:
	rm -f $(ALLOBJ) $(PROG)

depend: 
	$(NVCC) $(NVCCFLAGS) -M $(ALLSRC) > make.dep






