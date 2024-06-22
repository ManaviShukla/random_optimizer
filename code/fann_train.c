#include "fann.h"
#include "unistd.h"
#include <string.h>

using namespace std;

void usage() 
{
  printf("fann_train - use libfann to train on file \n");
  printf("Usage: fann_train -t [training file] \n");
  printf("\n");
  printf("-t [=filename] :        file containing training examples in fann format\n");
  printf("-h               :        show this dialog\n");
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

  /*   // grab attributes from 2nd line: */
  /*   fgets(databuf,4000,fid); */
  /*   tok = strtok( databuf, " " ); */
  /*   i=1; */
  /*   input[i] = atoi(tok); */
  /*   while (tok != NULL)  */
  /*     { */
  /*       tok = strtok(NULL," "); */
  /*       if (tok != NULL) { */
  /* 	i++; */
  /* 	input[i] = atof(tok); */
  /*       } */
  /*     } */
  /*   return(i); */
}


extern char *optarg;
extern int  optind, opterr;

int main(int argc, char* argv[])
{
  char* training_file;
  
  int c;
  static char optstring[] = "ht:";
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
	  printf("Attempting to load %s\n",training_file);
	  break;

	case '?':
	  printf("missing or invalid parameter\n");
	}
    }
  if (optind < argc)
    printf("input overflow - last processed arg: %s\n",argv[optind]);
  
  if (argc < 3)
    usage();


  const unsigned int max_epochs = 50000; // 500000
  const unsigned int epochs_between_reports = 1000;
  const float desired_error = (const float) 0.0005; // 0.001

  // standard:
  int num_instances;
  int num_input;
  int num_output;

  read_header(training_file, &num_instances, &num_input, &num_output);

  printf("instances: %i || input: %i || output: %i\n",num_instances,num_input,num_output);

  const unsigned int num_layers = 3; //3
  const unsigned int num_neurons_hidden = 3; //3
  
  printf("Number of attributes detected: %i\n",num_input);
  printf("Training with %i neurons at input layer\n\n",num_input);

  struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

  // Adjust parameters for the network: 
  /* Notes on tuning:      (see: see: http://leenissen.dk/fann/html/files/fann_data-h.html#fann_train_enum)
     algorithm:  these two are good, quickprop & rprop for more advanced users
     -both are backprop, but batch only props once per epoch, so is more accurate but slower
     error: tanh is more aggressive at weighing high error, but maybe not good with incremental (or cascade)
  */


  //fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL); // FANN_TRAIN_INCREMENTAL FANN_TRAIN_BATCH
  fann_set_learning_momentum(ann, 0.001);
  fann_set_learning_rate(ann, 0.001);
  //fann_set_train_error_function(ann,FANN_ERRORFUNC_LINEAR); // FANN_ERRORFUNC_LINEAR FANN_ERRORFUNC_TANH
  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

  // run the damn thing:
  fann_train_on_file(ann, training_file, max_epochs, epochs_between_reports, desired_error);

  /* 
  // cascade style:
  struct fann_train_data *train_data;
  train_data = fann_read_train_from_file(training_file);
  struct fann *ann = fann_create_shortcut(2, fann_num_input_train_data(train_data), fann_num_output_train_data(train_data));
  uint max_neurons = 50000;
  uint neurons_between_reports = 100;
  fann_cascadetrain_on_file(ann, training_file, max_neurons, neurons_between_reports, desired_error);
  */

  char outbuf[100];
  sprintf(outbuf, "%s_trained.net", training_file); // name based on training
  printf("Saving network to %s\n",outbuf); // name based on training
  fann_save(ann, outbuf);

  fann_destroy(ann);

  return 0;
}
