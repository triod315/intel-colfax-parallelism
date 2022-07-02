#include <mkl.h>
#include "distribution.h"


//vectorize this function based on instruction on the lab page
int diffusion(const int n_particles, 
              const int n_steps, 
              const float x_threshold,
              const float alpha, 
              VSLStreamStatePtr rnStream) {
  int n_escaped = 0;
  
  float x[n_particles];
  x[0:n_particles] = 0.0f;

  float rn[n_particles];
 
  vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, n_particles, rn, -1.0, 1.0);

  for (int i = 0; i < n_particles; i++) {
    for (int j = 0; j < n_steps; j++) {
      x[j] += dist_func(alpha, rn[j]); 
    }
    //if (x > x_threshold) n_escaped++;
  }

  for (int i = 0; i< n_steps; i++)
    if (x[i] > x_threshold)
      n_escaped++;

  return n_escaped;
}
