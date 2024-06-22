#include "fann.h"
#include "unistd.h"
#include <stdlib.h>
#include <string.h>
#include "fann_weval.c"


#define RANDN(N,M) N + ((M-N) * (rand() / ((double)RAND_MAX + 1))) // random # between N&M
#define MSE_MAX 999999
#define TEMP_INIT 0.1 // toasty
#define COOLING_CONSTANT 0.99 // for exponential decay
#define USE_GREEDY

/* Attempts to train a neural network using randomized hill climbing
   rather than backprop to identify optimal weights.  Loads basic network
   parameters from the training file, and calls fann_weval to evaluate a
   set of weights
*/

using namespace std;

void write_gnuplot_dat(char* datfile, int round, float var)
{
  FILE *fp = fopen(datfile, "a");
  fprintf(fp,"%i \t %f\n", round, var);
  fclose(fp);
}

void usage() 
{
  printf("fann_RHC_train  - Attempts to train a neural network using randomized hill climbing \n");
  printf("Usage: fann_RHC_train -t <training file>\n");
  printf("\n");
  printf("-t [=filename] :        file containing training examples in fann format\n");
  printf("-h              :        show this dialog\n");
  exit(0);  
}

void read_header(char* filename, int* num_instances, int* num_input, int* num_output)
{
  FILE* fid;
  fid = fopen(filename,"r");
  fann_type input[1000]; // size of input
  char *tok = NULL;    
  char databuf[4000]; // accomodate up to 1000 attributes (i think)
  int i;

  // read from 1st line of training file:
  fgets(databuf,4000,fid);

  // parse for input on spaces
  *num_instances = atoi(strtok(databuf, " "));
  *num_input = atoi(strtok(NULL, " "));
  *num_output = atoi(strtok(NULL, " "));
  fclose(fid);
}

extern char *optarg;
extern int  optind, opterr;

int main(int argc, char* argv[])
{

  int c;
  char* training_file;
  char* test_file;
  static char optstring[] = "ht:T:";
  opterr=0;

  while ((c=getopt(argc, argv, optstring)) != -1)
    {
      switch(c)
	{
	case 'h':
	  usage();
	  break;
	case 't':
	  training_file = optarg;
	  printf("Using training examples from %s \n",training_file);
	  break;
	case 'T':
	  test_file = optarg;
	  printf("Using test examples from %s \n",test_file);
	  break;
	case '?':
	  printf("missing or invalid parameter\n");
	}
    }
  if (optind < argc)
    printf("input overflow - last processed arg: %s\n",argv[optind]);

  if (argc < 3)
    usage();

  // default params
  const unsigned int num_layers = 3; //3
  const unsigned int num_neurons_hidden = 3; //3
  int num_instances;
  int num_input;
  int num_output;

  char outbuf[100];
  sprintf(outbuf, "%s_trained.net", training_file); // name based on training

  // create network
  read_header(training_file, &num_instances, &num_input, &num_output);
  struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

  // read in training data from file:
  struct fann_train_data *data;
  data = fann_read_train_from_file(training_file);

  float cthresh=0.0001; // termination criterion
  float MSE=MSE_MAX;  // returned from fann_weval - our generic cost estimate
  float bestMSE=MSE; // container for best score so far
  int round = 0; // counter
  int total_connections = fann_get_total_connections(ann);
  int total_neurons = fann_get_total_neurons(ann);
  float stepsize = 0.01; // for batch algorithm - the amount to try changing weights each round
  float Temp = TEMP_INIT;
  struct fann_connection *init_connections; // stores start configuration for each round
  
  init_connections = (struct fann_connection*) malloc(sizeof(struct fann_connection) * total_connections); // gotta allocate this space now and hope the net doesn't change
  fann_get_connection_array(ann,init_connections); // initialize to starting weights
  
  // stuff for greedy needs to be outside scope to work (simply)
  int be_greedy = 0;
  int some_connection; 
  int sign = 1;
  int greedy_steps=0;
  int back_moves=0;

  // loop until cost is below some threshold
  while (MSE > cthresh) {

    // save a copy of current weights in case we screw them up
    fann_get_connection_array(ann,init_connections);


    /* --> Simmulated Annealing algorithm overview <--
       1) Take a random step
       2) if step is better, take it (optional greedy step to keep the ball rolling)
       3) if step is worse, take it anyway with probability [ cost_improvement/Temperature ] 
       4) decrease temperature over time
    */


#ifdef USE_GREEDY
    const char* g="greedy";
    // Randomly change a weight by some quantity
    if (be_greedy==1){
      fann_set_weight(ann,init_connections[some_connection].from_neuron,init_connections[some_connection].to_neuron,init_connections[some_connection].weight+(sign*stepsize)); // keep a good thing going
      //printf("being greedy!\n");
    }
    else {
      sign = RANDN(0,1) > 0.5 ? 1 : -1; // produces 1 or -1
      some_connection = RANDN(1,total_connections);
      fann_set_weight(ann,init_connections[some_connection].from_neuron,init_connections[some_connection].to_neuron,init_connections[some_connection].weight+(sign*stepsize));
    }

#endif
    
#ifndef USE_GREEDY
    const char* g="notgreedy";
    sign = RANDN(0,1) > 0.5 ? 1 : -1; // produces 1 or -1
    some_connection = RANDN(1,total_connections);
    fann_set_weight(ann,init_connections[some_connection].from_neuron,init_connections[some_connection].to_neuron,init_connections[some_connection].weight+(sign*stepsize));
#endif
    
    // evaluate
    MSE = fann_weval(ann, data, num_instances, total_connections); 

    // if better, update score, else roll back unless prob < exp(- cost_difference / Temp)
    if (MSE < bestMSE){
      bestMSE=MSE;
      be_greedy=1;
      greedy_steps++;
    }
    else {
      be_greedy=0;
      if (RANDN(0,1) < (exp(-(MSE-bestMSE)/Temp))) {	
	bestMSE=MSE; // go anyway
	//printf("going anyway\n");
	back_moves++;
      }
      else
	fann_set_weight_array(ann,init_connections,total_connections);
    }
    
    Temp = Temp * COOLING_CONSTANT;
    
    

    if (round % 1000 == 0){
      char datnamebuf[100];
      // save for gnuplot
      sprintf(datnamebuf,"SA_converge_TI%1.1f_CC%1.5f_%s.dat",TEMP_INIT,COOLING_CONSTANT,g);
      write_gnuplot_dat(datnamebuf,round,(float)bestMSE);

      /* ACK!  i couldn't figure out why "hop outs" kept increasing
	 over time, until I finally accepted that there's an
	 interaction between the delta cost and the temperature such
	 that the [stepsize] changes in weights for a network that's
	 nearly perfectly trained have very little cost difference,
	 and are thus quite likely to be jumped out of.  SO: while
	 temperature would produce the expected effect if we were
	 picking a random weight each time, it won't in this scenario
	 because the cost differences shrink over time
      printf("delta cost: %f\n",MSE-bestMSE);
      printf("rand,prob = %f,%f\n",RANDN(0,1),(exp(-(MSE-bestMSE)/Temp)));
      */
      printf("%i: MSE=%1.6f, Temp=%f, P(.001)=%f,  #greedy=%d, #backsteps=%d   (progress saved)\n",round,bestMSE,Temp,(exp(-0.001/Temp)),greedy_steps,back_moves);
      greedy_steps = 0;
      back_moves = 0;
      fann_save(ann, outbuf);
    }
  
    round++;
  }

  printf("Error reached threshold (%f): %f\n",cthresh,MSE);

  printf("Saving network to %s\n",outbuf); // name based on training
  fann_save(ann, outbuf);

  fann_destroy(ann);

  return 0;
}
