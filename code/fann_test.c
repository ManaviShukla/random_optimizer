#include "unistd.h"
#include <stdio.h>
#include "floatfann.h"
#include <string.h>

using namespace std;

void usage() 
{
  printf("fann_test - use libfann to test on file \n");
  printf("Usage: fann_test -T file\n");
  printf("\n");
  printf("-T [=filename] :        file containing test examples in fann format\n");
  printf("-n [=filename] :        file containing a trained fann network\n");
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
  char* test_file;
  char* net_file;
  static char optstring[] = "hn:T:";
  opterr=0;

  while ((c=getopt(argc, argv, optstring)) != -1)
    {
      switch(c)
	{
	case 'h':
	  usage();
	  break;
	case 'T':
	  test_file = optarg;
	  printf("Using test examples from %s\n",test_file);
	  break;
	case 'n':
	  net_file = optarg;
	  printf("Loading network from %s\n",net_file);
	  break;
	case '?':
	  printf("missing or invalid parameter\n");
	}
    }
  if (optind < argc)
    printf("input overflow - last processed arg: %s\n",argv[optind]);

  if (argc < 2)
    usage();


  // Calculate Stuff:

  int num_instances;
  int num_input;
  int num_output;
  fann_type *net_out;
  int i,o;
  int correct=0;
  struct fann_train_data *data;
  struct fann *ann = fann_create_from_file(net_file);

  read_header(test_file, &num_instances, &num_input, &num_output);

  data = fann_read_train_from_file(test_file);

    for(i = 0; i < fann_length_train_data(data); i++)
    {
      net_out = fann_run(ann, data->input[i]);

      // Classification: take highest output      
      float max_net=0;
      int maxi_net=0;
      float max_actual = 0;
      int maxi_actual=0;

      for (o=0; o<num_output; o++) {
	if (net_out[o] > max_net) {
	  max_net = net_out[o];
	  maxi_net = o;
	}
	if (data->output[i][o] > max_actual) {
	  max_actual = data->output[i][o];
	  maxi_actual = o;
	}
      }

      const char *mark="-";
      if (maxi_net == maxi_actual) {
	correct++;
	mark="+";
      }

      // generate a nice report of each run:
      printf("%s%2i: [", mark, i);
      for (int o=0; o<num_output; o++) {
	printf("%f ",net_out[o]);
      }
      printf("],[");
      for (int o=0; o<num_output; o++) {
	printf("%f ",data->output[i][o]);
      }
      printf("] (network out, actual)\n");

    }
  
  printf("\n Number correct = %i\n",correct);
  printf("Instances = %i\n",num_instances);
  printf("Percent correct = %f\n",100*(float)correct/(float)num_instances);

  fann_destroy(ann);
  return 0;
}
