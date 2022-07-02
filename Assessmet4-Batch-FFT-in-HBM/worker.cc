#include <mkl.h>
#include <hbwmalloc.h>

//implement scratch buffer on HBM and compute FFTs, refer instructions on Lab page
void runFFTs( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle) {
  MKL_Complex8* buffer;
  hbw_posix_memalign((void **) &buffer, 4096, sizeof(MKL_Complex8)*fft_size);
  for(size_t i = 0; i < num_fft; i++) {
    #pragma omp parallel for 
    for (size_t j = 0; j < fft_size; j++)
    {
	buffer[j].real = data[j + i*fft_size].real;
	buffer[j].imag = data[j + i*fft_size].imag;
    }

    DftiComputeForward(*fftHandle, &buffer[0]);
    #pragma omp parallel for
    for (size_t j = 0; j < fft_size; j++) {
    	data[j + i*fft_size].real = buffer[j].real;
	data[j + i*fft_size].imag = buffer[j].imag;
    }
  }
  hbw_free(buffer);
}
