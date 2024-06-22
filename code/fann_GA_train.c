#include "fann.h"
#include "unistd.h"
#include <stdlib.h>
#include <string.h>
#include "fann_weval.c"


#define RANDNM(N,M) N + ((M-N) * (rand() / ((double)RAND_MAX + 1))) // random # between N&M
#define MSE_MAX 999999
#define WMIN -50
#define WMAX 50
#define POPSIZE 100
#define FECUNDITY 0.1 // proportion of population to crossover (and kill) each generation 
#define MUTATIONRATE 0.6  // proportion of updates where weights get randomized 
#define MUTATIONCOUNT 30 // number of weights to mutate when it mutates
//#define SAMPLE_SEQ // sequential, otherwise random

using namespace std;




int find(const int vec[], int len, int item)
{
  for (int i = 0; i < len; ++i)
    {
      if (vec[i]==item)
	return i;
    }
  return -1;
}

void print_pdist(float dist[], int len)
{
  printf("[\n",len);
  for (int i = 0; i < len; ++i)
    {
      printf("%f\n",dist[i]);
    }
  printf("]\n");
}

void print_list(int* list, int len)
{
  printf("[\n");
  for (int i = 0; i < len; ++i)
    {
      printf("%i\n",list[i]);
    }
  printf("]\n");
}

void print_weights(struct fann_connection* connections, int num_cons)
{
  printf("weights=[\n");
  for (int i = 0; i < num_cons; ++i)
    {
      printf("%f ",connections[i].weight);
    }
  printf("]\n");
}

void write_gnuplot_dat(char* datfile, int round, float var)
{
  FILE *fp = fopen(datfile, "a");
  fprintf(fp,"%i \t %f\n", round, var);
  fclose(fp);
}

void write_gnuplot_dat(char* datfile, float var[], int len)
{
  FILE *fp = fopen(datfile, "a");
  for (int i = 0; i < len; ++i)
    {
      fprintf(fp,"%f\n", var[i]);      
    }
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

  // read params from training data
  read_header(training_file, &num_instances, &num_input, &num_output);

  // read in training data from file:
  struct fann_train_data *data;
  data = fann_read_train_from_file(training_file);




  // GA setup:
  float f = FECUNDITY; // rate of crossover for each generation (fraction of p)
  int p = POPSIZE; // size of population
  float m = MUTATIONRATE;

  struct fann *bestann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
  struct fann* ann_array[p];

  float cthresh=0.001; // termination criterion
  float MSE=MSE_MAX;  // returned from fann_weval - our generic cost estimate
  float bestMSE=MSE; // container for best score so far
  int round = 0; // counter
  int total_connections = fann_get_total_connections(bestann);
  int total_neurons = fann_get_total_neurons(bestann);

  struct fann_connection* bestconn =  (struct fann_connection*) malloc(sizeof(struct fann_connection) * total_connections);  // malloc the space need for the population of connections 
  struct fann_connection* tempconn1 =  (struct fann_connection*) malloc(sizeof(struct fann_connection) * total_connections);  // malloc the space need for the population of connections 
  struct fann_connection* tempconn2 =  (struct fann_connection*) malloc(sizeof(struct fann_connection) * total_connections);  // malloc the space need for the population of connections 

  // loop through num_individuals to randomize initial popluation weights (don't forget to figure this out empirically from SA/TRAIN) 
  for (int i = 0; i < p; ++i)
    {
      ann_array[i] = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
      fann_randomize_weights(ann_array[i],WMIN,WMAX); 
    }


  // loop until cost is below some threshold
  while (MSE > cthresh) {

    int r = f*p; // num individuals needed in each list

    /* --> Genetic algorithm overview <--
       PARAMS: 
       p: number of networks in the population
       r: number of individuals to replace by crossover in each generation
       m: mutation rate

       0) produce random weights for p populations of networks
       1) evaluate all networks in first generation, and assign each a 
       probability (of being sampled) in proportion with their cost.  
       2) sample f*p/2 PAIRS of networks according to probability
       3) randomly swap n/2 weights across each pair, and mutate m percent (randomize a weight)
       4) add new generation to p
       5) optional: sample some proportion k of p via (1-prob) to "kill"
       6) evaluate all NEW networks, and repeat

       ** population will have to start with enough individuals to
       ** cover the possible desirable variations in weights.
       ** Crossover simply involves weight swapping, and mutation
       ** simply re-randomizing (within a reasonable interval).  The
       ** population will grow over time and may get unwieldy for
       ** memory usage, but computation shouldn't matter because only
       ** new networks are evaluated
       */

    // loop through population and assign a cost to each by calling weval

    /*
      NOTE: because GA's tend to homogenize your population, the cost
      distribution tends to become uniform.  Thus, sampling from it tends to
      miss the incremental improvements that come with a good mutation or
      crossover, since everything is near chance.  One solution is to make
      the selection greedier, as seen below.
    */
    float cost_array[p];
    float cost_sum = 0;
    for (int i = 0; i < p; ++i)
      {
	MSE = fann_weval(ann_array[i], data, num_instances, total_connections); 
	cost_array[i] = MSE;
	cost_sum += MSE;
	if (MSE<bestMSE) {
	  bestMSE = MSE; // mark lowest cost while we're at it
	  fann_get_connection_array(ann_array[i], bestconn); // save weights
	}
      }


    int crossoverlist[r]; // array listing the members to cross this round
    int slaughterlist[r]; // array listing the members to replace this round


#ifndef SAMPLE_SEQ
    const char* ss="random";
    // Sample from probability distributions over cost - convert to probability distro over cost
    float cost_prob[p];
    for (int i = 0; i < p; ++i)
      {
	cost_prob[i] = cost_array[i]/cost_sum;
      }
    //printf("cost array"); print_pdist(cost_array,p);
    //printf("cost_distribution="); print_pdist(cost_prob,p);

    // sample conns by cost until we have f*p of them for killin'
    int nsampled = 0; // counter for number of doods sampled
    int randi; // random individual to try sampling

    while (nsampled < r){
      randi = RANDNM(0,p); // random index of individual in population
      if (RANDNM(0,1) < cost_prob[randi]) {
	if (find(slaughterlist, sizeof(slaughterlist)/sizeof(float), randi)==-1) { // sampling w/out replacement 
	  slaughterlist[nsampled] = randi; // index of "breeding" member of ann_array
	  nsampled++;
	}
      }
    }
    //printf("slaughterlist="); print_list(slaughterlist,sizeof(slaughterlist)/sizeof(float));

    // Sample conns by fitness to find best individuals for crossover
    /*     NOTE: Although each list is sampled w/out replacement, the two */
    /*     are sampled independently and can therefore overlap.  In this */
    /*     respect the alg behaves like the one in Mitchell, where some */
    /*     individuals are sampled by fitness and the rest sampled again */
    /*     by fitness to participate in crossover (meaning some of the */
    /*     parents can get replaced by offspring, just as the case here)  */
     



    // Easy way - just sample again based on cost (p-fp) times and take what's left to be most fit
    nsampled = 0; // counter for number of doods sampled
      int reject[p-r];   
      while (nsampled < (p-r)){
	randi = RANDNM(0,p);  // random index of individual in population
	if (RANDNM(0,1) < cost_prob[randi]) {
	  if (find(reject, sizeof(reject)/sizeof(float), randi)==-1) { // sampling w/out replacement 
	    reject[nsampled] = randi; // index of "breeding" member of ann_array
	    nsampled++;
	  }
	}
      }

      int match=0; 
      for (int i = 0; i < p; ++i)
	{
	  if (find(reject,sizeof(reject)/sizeof(float), i)==-1){
	    crossoverlist[match]=i;
	    match++;
	  }
	}
    //printf("crossoverlist="); print_list(crossoverlist,sizeof(crossoverlist)/sizeof(float));
    
    /* - hard way - try to invert cost probability function to represent fitness and sample from that
    // now sample an inverted cost distro for members to crossover (alternatively, could sample up to (p - f*p) an take the leftovers as best)
    // -> need a new prob distro describing fitness - (2*mean - cost_array[i]) should do it (don't forget to renorm in case a member was greater than 2/p)
    float fitness_prob[p];
    int crossoverlist[r];
    float fitsum = 0;
    float fitness;
    // invert prob distro
    for (int i = 0; i < p; ++i)
    {
    printf("fitness = %f\n", fitness);
    fitness = 2/p - cost_prob[i];
    fitness_prob[i] = fitness > 0 ? fitness : 0;
    fitsum += fitness_prob[i]; // for renorming
    }
    printf("fitprob 1= %f\n",fitness_prob[1]);
    // renorm it
    for (int i = 0; i < p; ++i)
    {
    fitness_prob[i] = fitness_prob[i]/fitsum;
    }

    printf("fitsum = %f\n",fitsum);
    printf("fitness distribution"); print_pdist(fitness_prob,p);

    // sample from it for crossover members
    nsampled = 0;
    while (nsampled < f*p){
    randconn = RANDNM(0,total_neurons);
    if (RANDNM(0,1) > fitness_prob[randconn]) {
    if (find(crossoverlist,sizeof(crossoverlist)/sizeof(float),randconn)==0) { // sampling w/out replacement 
    crossoverlist[nsampled] = randconn; // index of "breeding" member of ann_array
    nsampled++;
    }
    }
    }
    printf("crossover list"); print_list(crossoverlist,sizeof(crossoverlist)/sizeof(float));
    */
#endif

#ifdef SAMPLE_SEQ
    // OR a greedier approach - select the f*p best and worst individuals for each list
      const char* ss="sequential";
      float min;
      float max;
      int mini;
      int maxi;
      for (int i=0; i<r; ++i) { // loop through crossover population 
	min = MSE_MAX;
	max = 0;

	for (int j=0; j<p; ++j) { // loop through whole population
	  if (find(crossoverlist,i,j)==-1) { // don't include previous mins in calculation
	    if (min>cost_array[j]) {
	      min = cost_array[j];
	      mini = j;
	    }
	  }

	  if (find(slaughterlist,i,j)==-1) { // don't include previous maxes in calculation
	    if (max<cost_array[j]) {
	      max = cost_array[j];
	      maxi = j;
	    }
	  }
	}
	//printf("min = %f\n",min);
	//printf("max = %f\n",max);
	crossoverlist[i] = mini;
	slaughterlist[i] = maxi;
      }
      //printf("cost_array="); print_pdist(cost_array,p);
      //printf("crossoverlist="); print_list(crossoverlist,sizeof(crossoverlist)/sizeof(float));
      //printf("slaughterlist="); print_list(slaughterlist,sizeof(slaughterlist)/sizeof(float));
#endif
    



    // go pairwise through crossoverlist and do random crossover
    for (int i = 0; i < r; i+=2)
      {
	// lazy crossover - sample with replacement (rather than swapping exactly total_connections/2)
	// read out weights of the fittest samples into temporary connection structs:
	fann_get_connection_array(ann_array[crossoverlist[i]],tempconn1);
	fann_get_connection_array(ann_array[crossoverlist[i+1]],tempconn2);

	/* 	printf("weights before cross\n"); */
	/* 	print_weights(tempconn1, total_connections); */
	/* 	print_weights(tempconn2, total_connections); */

	float w1,w2;
	int connsamp;
	for (int j = 0; j < total_connections/2; ++j) // cross up to n/2 weights (depending on sampling)
	  {
	    // sample a connection in set {0,total_connections} and swap weight between nets
	    connsamp = RANDNM(0,total_connections);
	    w1 = tempconn1[connsamp].weight;
	    w2 = tempconn2[connsamp].weight;
	    tempconn1[connsamp].weight = w2;
	    tempconn2[connsamp].weight = w1;
	  }
	// write these two new "children" to the location of some less fit individuals
	fann_set_weight_array(ann_array[slaughterlist[i]], tempconn1, total_connections);
	fann_set_weight_array(ann_array[slaughterlist[i+1]], tempconn2, total_connections);
	
	/* 	printf("weights after cross\n"); */
	/* 	print_weights(tempconn1, total_connections); */
	/* 	print_weights(tempconn2, total_connections); */
	//	return 0;
      }

    // Randomly "mutate" weights according to rate m
    for (int i = 0; i < p; ++i)
      {
	if (RANDNM(0,1) < m)
	  for (int j = 0; j < MUTATIONCOUNT; ++j)
	    {
	      fann_set_weight(ann_array[i], (int)RANDNM(0,total_neurons),(int)RANDNM(0,total_neurons),RANDNM(WMIN,WMAX));	      
	    }
      }
    
    char datnamebuf[100];

    // Uncomment to save/report snapshots of the cost array over the whole population
    /*     if (round % 5000 == 0) { */
    /*       //printf("cost_array="); print_pdist(cost_array,p); // Uncomment for a matlab friendly cost array over all individuals */
    /*       sprintf(datnamebuf,"costarr%i_p%i_f%1.1f_mr%1.1f_mc%i.dat",round,p,f,m,MUTATIONCOUNT); */
    /*       write_gnuplot_dat(datnamebuf,cost_array,p); */
    /*     } */

    // wipe out datfile for these params
    if (round == 0) {
      sprintf(datnamebuf,"GA_converge_p%i_f%1.1f_mr%1.1f_mc%i_%s.dat",p,f,m,MUTATIONCOUNT,ss);
      FILE *fp = fopen(datnamebuf, "w");
      fprintf(fp,"# GA converge data for \n");
      fclose(fp);
    }

    if (round % 250 == 0){
      // Uncomment for cost report of slaughter and crossover lists:
      /*       for (int i = 0; i < r; ++i) { */
      /* 	printf("slaughtered unit cost: %f ; crossed unit cost: %f ; diff: %f \n",cost_array[slaughterlist[i]],cost_array[crossoverlist[i]],cost_array[slaughterlist[i]]-cost_array[crossoverlist[i]]); */
      /*       } */

      /*       printf("cost_distribution="); print_pdist(cost_prob,p); */
      /*       printf("slaughterlist="); print_list(slaughterlist,sizeof(slaughterlist)/sizeof(float)); */
      /*       printf("crossoverlist="); print_list(crossoverlist,sizeof(crossoverlist)/sizeof(float)); */
      
      printf("round=%i\n",round);
      printf("MSE=%1.7f\n\n",bestMSE);
      
      // save for gnuplot
      sprintf(datnamebuf,"GA_converge_p%i_f%1.1f_mr%1.1f_mc%i_%s.dat",p,f,m,MUTATIONCOUNT,ss);
      write_gnuplot_dat(datnamebuf,round,(float)bestMSE);

      fann_set_weight_array(bestann, bestconn, total_connections);
      fann_save(bestann, outbuf);
    }

    round++;
 
  }

  printf("MSE reached threshold (%f): %f\n",cthresh,MSE);

  printf("Saving network to %s\n",outbuf); // name based on training
  fann_set_weight_array(bestann, bestconn, total_connections);
  fann_save(bestann, outbuf);
  
  
  fann_destroy(bestann);
  for (int i = 0; i < p; ++i)
    {
      fann_destroy(ann_array[i]);
    }

  return 0;
}

/*  LocalWords:  PARAMS
 */
