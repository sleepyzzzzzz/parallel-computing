#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unistd.h>
#include <numa.h>

#define STATIC_CHUNK 10

void usage(const char *name)
{
  std::cout << "usage: " << name
            << " matrix-size nworkers"
            << std::endl;
  exit(-1);
}

/* 
  initialize matrixes:
  matrix A: nxn, uniform random number
  P: a compact representation of a permutation matrix
  U: the upper-triangular matrix - same value as A
  L: the lower-triangular matrix - 1s on the diagonal and 0s above the diagonal
*/
void random_number_generate(double **A, int *P, double **L, double **U, int n, int nworkers)
{
  // each thread takes one certain rows using 1D block-cyclic distribution
  #pragma omp parallel num_threads(nworkers) shared(A, P, L, U) firstprivate(n, nworkers)
  {
    int tid = omp_get_thread_num();
    struct drand48_data randBuffer;
    srand48_r((tid + 1) * 51961471, &randBuffer);
    for (int i = tid; i < n; i += nworkers)
    {
      A[i] = (double *)numa_alloc_local(n * sizeof(double));
      U[i] = (double *)numa_alloc_local(n * sizeof(double));
      L[i] = (double *)numa_alloc_local(n * sizeof(double));
      memset(L[i], 0, n * sizeof(double));
      L[i][i] = 1;
      P[i] = i + 1;
      for (int j = 0; j < n; j++)
      {
        drand48_r(&randBuffer, &A[i][j]);
        U[i][j] = A[i][j];
      }
    }
  }
}

// perform LU decomposition with partial pivoting
void ludecomposition(double **A, int *P, double **L, double **U, int n, int nworkers)
{
  double global_max = 0.00;
  int global_k_prime = 0;
  int tmp_k = 0;
  double *atmp_k = new double[n];
  double *ltmp_k = new double[n];
  int tmp_k_prime = 0;
  double *atmp_k_prime = new double[n];
  double *ltmp_k_prime = new double[n];

  #pragma omp parallel num_threads(nworkers) shared(A, P, L, U, global_max, global_k_prime, tmp_k, atmp_k, ltmp_k, tmp_k_prime, atmp_k_prime, ltmp_k_prime) firstprivate(n, nworkers)
  {
    double local_max = 0.00;
    int local_k_prime = 0;
    int start_row = 0;

    int tid = omp_get_thread_num();
    for (int k = 0; k < n; k++)
    {
      // get the first row the thread takes on start with k
      start_row = tid + k / nworkers * nworkers;
      if (k % nworkers <= tid) {
        start_row = tid + k / nworkers * nworkers;
      }
      else {
        start_row = tid + (k / nworkers + 1) * nworkers;
      }
      // get the maximum values among rows for the thread for column k start with row k 
      for (int i = start_row; i < n; i += nworkers)
      {
        if (local_max < abs(U[i][k]))
        {
          local_max = abs(U[i][k]);
          local_k_prime = i;
        }
      }
      // get the global maximum value and its row index
      #pragma omp critical (update_globals)
      {
        if (local_max > global_max)
        {
          global_max = local_max;
          global_k_prime = local_k_prime;
        }
        local_max = 0.00;
        local_k_prime = 0;
      }

      #pragma omp barrier
      #pragma omp single
      {
        // check if the matrix is a singular matrix
        if (global_max == (double) 0)
        {
          printf("error: singular matrix\n");
          exit(0);
        }
        global_max = 0.00;
      }

      /*
        swap P[k] and P[k']
        swap U(k,:) and U(k',:)
        swap l(k,1:k-1) and l(k',1:k-1)
      */
      if (tid == k % nworkers) {
        tmp_k = P[k];
        memcpy(atmp_k, U[k], n * sizeof(double));
        memcpy(ltmp_k, L[k], k * sizeof(double));
      }

      if (tid == global_k_prime % nworkers) {
        tmp_k_prime = P[global_k_prime];
        memcpy(atmp_k_prime, U[global_k_prime], n * sizeof(double));
        memcpy(ltmp_k_prime, L[global_k_prime], k * sizeof(double));
      }

      #pragma omp barrier
      if (tid == global_k_prime % nworkers) {
        P[global_k_prime] = tmp_k;
        memcpy(U[global_k_prime], atmp_k, n * sizeof(double));
        memcpy(L[global_k_prime], ltmp_k, k * sizeof(double));
      }

      if (tid == k % nworkers) {
        P[k] = tmp_k_prime;
        memcpy(U[k], atmp_k_prime, n * sizeof(double));
        memcpy(L[k], ltmp_k_prime, k * sizeof(double));
      }

      // get the first row the thread takes on start with k + 1
      if ((k + 1) % nworkers <= tid) {
        start_row = tid + (k + 1) / nworkers * nworkers;
      }
      else {
        start_row = tid + ((k + 1) / nworkers + 1) * nworkers;
      }

      #pragma omp barrier
      for (int i = start_row; i < n; i += nworkers)
      {
        L[i][k] = U[i][k] / U[k][k];
        U[i][k] = 0;
        for (int j = k + 1; j < n; j++) {
          U[i][j] = U[i][j] - L[i][k] * U[k][j];
        }
      }
    }
  }
  delete[] atmp_k;
  delete[] ltmp_k;
  delete[] atmp_k_prime;
  delete[] ltmp_k_prime;
}

/* 
  verify PA = LU
  compute the sum of Euclidean norms of the columns of the residual matrix - PA -LU
*/
void verification(double **A, int *P, double **L, double **U, int n, int nworkers)
{
  double diff = 0.00;
  #pragma omp parallel for schedule (static, STATIC_CHUNK) shared(diff)
  for (int i = 0; i < n; i++)
  {
    double col_diff = 0.00;
    for (int j = 0; j < n; j++)
    {
      double tmp = A[P[i] - 1][j];
      for (int k = 0; k < n; k++)
      {
        tmp = tmp - L[i][k] * U[k][j];
      }
      col_diff = col_diff + tmp * tmp;
    }
    #pragma omp critical
    {
      diff = diff + sqrt(col_diff);
    }
  }
  printf("diff: %f\n", diff);
}

int main(int argc, char **argv)
{
  const char *name = argv[0];

  if (argc < 3)
    usage(name);

  int matrix_size = atoi(argv[1]);

  int nworkers = atoi(argv[2]);

  std::cout << name << ": "
            << matrix_size << " " << nworkers
            << std::endl;

  omp_set_num_threads(nworkers);

  // allocate memory for matrix A, P, L, U
  double **A = (double **)numa_alloc_local(matrix_size * sizeof(double*));
  int *P = (int *)numa_alloc_local(matrix_size * sizeof(int));
  double **L = (double **)numa_alloc_local(matrix_size * sizeof(double*));
  double **U = (double **)numa_alloc_local(matrix_size * sizeof(double*));

  random_number_generate(A, P, L, U, matrix_size, nworkers);

  auto start = std::chrono::steady_clock::now();

  ludecomposition(A, P, L, U, matrix_size, nworkers);

  auto end = std::chrono::steady_clock::now();
  auto diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  printf("elapsed time: %ds\n", diff);

  verification(A, P, L, U, matrix_size, nworkers);

  // free allocated memory
  #pragma omp parallel num_threads(nworkers) shared(A, P, L, U) firstprivate(matrix_size, nworkers)
  {
    int tid = omp_get_thread_num();
    #pragma omp for schedule (static)
    for (int i = tid; i < matrix_size; i += nworkers) {
      if (A[i] != NULL)
      {
        numa_free(A[i], matrix_size * sizeof(double));
      }
      if (L[i] != NULL) {
        numa_free(L[i], matrix_size * sizeof(double));
      }
      if (U[i] != NULL) {
        numa_free(U[i], matrix_size * sizeof(double));
      }
    }
  }
  numa_free(A, matrix_size * sizeof(double*));
  numa_free(P, matrix_size * sizeof(int));
  numa_free(L, matrix_size * sizeof(double*));
  numa_free(U, matrix_size * sizeof(double*));
  return 0;
}
