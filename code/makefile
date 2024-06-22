CC=g++
CFLAGS=-g
PREFIX := /usr/local

.PHONY: default install clean uninstall

default: all

all: fann_train fann_test fann_profile  fann_RHC_train fann_SA_train fann_GA_train fann_weightrange

fann_train: fann_train.o
	$(CC) -o $@ $< -lfann

fann_train.o: fann_train.c
	  $(CC) $(CFLAGS) -c $<

fann_test: fann_test.o
	$(CC) -o $@ $< -lfann

fann_test.o: fann_test.c
	  $(CC) $(CFLAGS) -c $<

fann_profile: fann_profile.o
	$(CC) -o $@ $< -lfann

fann_profile.o: fann_profile.c
	  $(CC) $(CFLAGS) -c $<

fann_RHC_train.o: fann_RHC_train.c
	  $(CC) $(CFLAGS) -c $<

fann_RHC_train: fann_RHC_train.c 
	  $(CC) $(CFLAGS) -o $@ $< -lfann

fann_SA_train.o: fann_SA_train.c
	  $(CC) $(CFLAGS) -c $<

fann_SA_train: fann_SA_train.c 
	  $(CC) $(CFLAGS) -o $@ $< -lfann

fann_GA_train.o: fann_GA_train.c
	  $(CC) $(CFLAGS) -c $<

fann_GA_train: fann_GA_train.c 
	  $(CC) $(CFLAGS) -o $@ $< -lfann

fann_weightrange.o: fann_weightrange.c
	  $(CC) $(CFLAGS) -c $<

fann_weightrange: fann_weightrange.o 
	  $(CC) $(CFLAGS) -o $@ $< -lfann

clean: 
	rm -vf *.o *~ fann_train fann_test fann_profile fann_RHC_train fann_SA_train fann_GA_train fann_weightrange