#include <fann.h>

int main(int argc, char *argv[])
{
  const char* netfile = "voxpat3_disc_train.fann_trained.net";
  struct fann *ann = fann_create_from_file(netfile);
  float max = -99999999;
  float min = 99999999;
  int total_connections = fann_get_total_connections(ann);

  struct fann_connection* connections =  (struct fann_connection*) malloc(sizeof(struct fann_connection) * total_connections);
  fann_get_connection_array(ann,connections); 

  for (int i = 0; i < fann_get_total_connections(ann); ++i)
  {
    if (connections[i].weight < min)
      min = connections[i].weight;
    if (connections[i].weight > max)
      max = connections[i].weight;

    printf("weight %d = %f\n",i,connections[i].weight);

 /*    printf("Min weight = %f\n",min); */
/*     printf("Max weight = %f\n",max); */
  }
	    
  printf("Min weight = %f\n",min);
  printf("Max weight = %f\n",max);

  return 0;
}
	
