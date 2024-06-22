#include "fann.h"

float fann_weval(struct fann *ann, struct fann_train_data *data, int num_instances, int total_connections) 
{
  float MSE;

  // option 1) loop over data and calculate MSE

  fann_reset_MSE(ann); // MSE seems to accumulate over time

  //printf("init MSE=%f\n",MSE);
  //MSE = fann_get_MSE(ann);
  //printf("starting MSE=%f\n",MSE);

  int i;
  for(i = 0; i < num_instances; i++) // or use fann_length_train_data(data)
    {
      fann_test(ann, data->input[i], data->output[i]);
    }

  MSE = fann_get_MSE(ann);
  //printf("updated MSE = %f",MSE);

  // option 2) loop over data and calculate percent correct



  return MSE;
}
