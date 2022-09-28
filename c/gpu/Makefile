include make.inc

CUSRX = fd.cu interpolate.cu paste.cu relax.cu \
		restrict.cu cut.cu translate.cu lfc.cu \
		ext-potential.cu mblas.cu linalg.cu \
		elementwise.cu
CUSRC = $(addprefix cuda/, $(CUSRX))

CSRC  = setup.c bc.c blas.c operators.c transformers.c

CUGPAWLIB = libgpaw-cuda.a

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

cuda/%.o: cuda/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

