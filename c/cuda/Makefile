include make.inc

CUSRC = fd-cuda.cu interpolate-cuda.cu paste-cuda.cu relax-cuda.cu \
		restrict-cuda.cu cut-cuda.cu translate-cuda.cu lfc-cuda.cu \
		ext-pot-cuda.cu mblas-cuda.cu blas-cuda.cu linalg-cuda.cu

CSRC  = gpaw-cuda.c operators-cuda.c bc-cuda.c transformers-cuda.c


all: $(CUGPAWLIB)

ALLSRC = $(CSRC) $(CUSRC)
ALLOBJ = $(CSRC:.c=.o) $(CUSRC:.cu=.o)

-include make.dep

.PHONY: depend clean

$(CUGPAWLIB): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f $(ALLOBJ) $(CUGPAWLIB)

depend:
	$(NVCC) $(NVCCFLAGS) -M $(ALLSRC) > make.dep

.c.o:
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

