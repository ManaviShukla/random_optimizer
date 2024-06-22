#include "fann.h"
#include "unistd.h"
#include <stdlib.h>
#include <string.h>
#include "fann_weval.c"


#define RANDN(N,M) N + ((M-N) * (rand() / ((double)RAND_MAX + 1))) // random # between N&M
#define MSE_MAX 999999

//#define DOALG1



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

  float cthresh=0.005; // termination criterion
  float MSE=MSE_MAX;  // returned from fann_weval - our generic cost estimate
  float bestMSE=MSE; // container for best score so far
  int round = 0; // counter
  int total_connections = fann_get_total_connections(ann);
  int total_neurons = fann_get_total_neurons(ann);
  float stepsize = 0.01; // for batch algorithm - the amount to try changing weights each round
  struct fann_connection *init_connections; // stores start configuration for each round
  
  init_connections = (struct fann_connection*) malloc(sizeof(struct fann_connection) * total_connections); // gotta allocate this space now and hope the net doesn't change
  fann_get_connection_array(ann,init_connections); // initialize to starting weights

  // loop until cost is below some threshold
  while (MSE > cthresh) {

    /* --> RHC algorithm overview <--
       Steps:
       1) load current weight configuration
       -- incremental/stochastic version --
       2) try a random change from the "move set" (possible changes to current config, which in this case is any weight change)
       3) evaluate performace
       4) if better, update network, if not, roll back 
       -if we haven't improved in [a while], do a random restart and try again
       -- batch/standard version --
       2) enumerate all possible changes in "move set"
       3) use cost function to find best move to take and take it
       4) if no moves improve our state, save weights, do a random restart, and try again

       ** weights are continuous values, so the moveset is tricky (b/c
       ** it's huge).  a half-way solution is to iterate through
       ** network and randomize each weight in turn (batch version).
       ** A smarter way might be to remeber what the last weight
       ** change was and try it again the other more gradient-like
       ** option is to have a stepsize (which can decay), and only
       ** allow moves +/- stepsize from each weight as the moveset
       */

    // save a copy of current weights in case we screw them up
    fann_get_connection_array(ann,init_connections);
    


    // Alg1) incremental-random:  make a random change to weights and look for improvement (no explicit gradient)
    #ifdef DOALG1
    const char* alg="incrm";
    fann_set_weight(ann,(int)RANDN(1,total_neurons),(int)RANDN(1,total_neurons),RANDN(-1,1));
    #endif


    // Alg2) batch-unitstep:  enumerate all weight changes using a fixed stepsize and pick the best (explicit gradient)
    //         --> manages all weight changes manually for speed (rather than using init_connections to reset)
    #ifndef DOALG1
    const char* alg="batch";
    int best_conn; // best connection to alter for this round
    float minSE=MSE_MAX;
    int best_sign=1; // which way to step for best move
    int i;
    for (i=0; i<total_connections; i++){
      fann_set_weight(ann,init_connections[i].from_neuron,init_connections[i].to_neuron,init_connections[i].weight+stepsize); // +stepsize
      MSE = fann_weval(ann, data, num_instances, total_connections);
      if (MSE<minSE) {
	minSE = MSE;
	best_conn = i;
	best_sign = 1;
      }
      
      fann_set_weight(ann,init_connections[i].from_neuron,init_connections[i].to_neuron,init_connections[i].weight-stepsize); // -stepsize
      MSE = fann_weval(ann, data, num_instances, total_connections);
      if (MSE<minSE) {
	minSE = MSE;
	best_conn = i;
	best_sign = -1;
      }

      fann_set_weight(ann,init_connections[i].from_neuron,init_connections[i].to_neuron,init_connections[i].weight); // -reset this connection to original weight
    }
    fann_set_weight(ann,init_connections[best_conn].from_neuron,init_connections[best_conn].to_neuron,init_connections[best_conn].weight+best_sign*stepsize); // apply best step we found

/*     //if we're stuck, restart */
/*     //fann_randomize_weights(ann, -1, 1); */
#endif

    // if better, update score, else roll back 
    MSE = fann_weval(ann, data, num_instances, total_connections); // evaluate
    if (MSE < bestMSE)
      bestMSE=MSE;
    else
      fann_set_weight_array(ann,init_connections,total_connections);

    
    char datnamebuf[100];
 
    if (round % 300 == 0){
      // save for gnuplot
      sprintf(datnamebuf,"RHC_converge_%s.dat",alg);
      write_gnuplot_dat(datnamebuf,round,(float)bestMSE);

      printf("round %i: best MSE=%f  (progress saved)\n",round,bestMSE);
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
