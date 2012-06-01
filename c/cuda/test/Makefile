include ../make.inc

CUSRC = 

CSRC  = kernel_test.c
PROG  = kernel_test


all: $(PROG)

ALLSRC = ../../bmgs/bmgs.c $(CSRC) $(CUSRC)

ALLOBJ = $(CSRC:.c=.o) $(CUSRC:.cu=.o)

NVLDFLAGS=-L.. -lgpaw-cuda -L$(PYTHONLIBdir) -lpython2.6 -lutil -L$(CUDALIBdir) -lcuda -lcublas

-include make.dep

.PHONY: depend clean


../$(CUGPAWLIB):
	cd .. ; make

$(PROG): $(ALLSRC) ../$(CUGPAWLIB)
	$(NVCC) $(NVCCFLAGS) $(NVLDFLAGS) $(ALLSRC) -o $@

clean:
	rm -f $(ALLOBJ) $(PROG)

depend: 
	$(NVCC) $(NVCCFLAGS) -M $(ALLSRC) > make.dep






