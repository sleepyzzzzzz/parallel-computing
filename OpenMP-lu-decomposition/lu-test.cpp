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

#define STATIC_CHUNK 5

int value = 0;

bool cmp(double a, double b)
{
  return abs(a) < abs(b);
}

long fib(int n)
{
  if (n < 2)
    return n;
  else
    return fib(n - 1) + fib(n - 2);
}

void usage(const char *name)
{
  std::cout << "usage: " << name
            << " matrix-size nworkers"
            << std::endl;
  exit(-1);
}

void print_matrix(double **A, int *P, double **L, double **U, int n)
{
  printf("A: \n");
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }
  printf("P: \n");
  for (int i = 0; i < n; i++)
  {
    printf("%d ", P[i]);
  }
  printf("\n");
  printf("L: \n");
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      printf("%f ", L[i][j]);
    }
    printf("\n");
  }
  printf("U: \n");
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      printf("%f ", U[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void random_number_generate(double **A, double **A1, int n, int nworkers)
{
  #pragma omp parallel num_threads(nworkers) shared(A, A1)
  {
    int tid = omp_get_thread_num();
    struct drand48_data randBuffer;
    srand48_r((tid + 1) * 51961471, &randBuffer);
    #pragma omp parallel for schedule (static)
    for (int i = tid; i < n; i += nworkers)
    {
      A[i] = (double *)numa_alloc_local(n * sizeof(double));
      A1[i] = (double *)numa_alloc_local(n * sizeof(double));
      #pragma omp parallel for schedule (static)
      for (int j = 0; j < n; j++)
      {
        drand48_r(&randBuffer, &A[i][j]);
        A[i][j] = A[i][j];
        A1[i][j] = A[i][j];
      }
    }
  }
}

void ludecomposition(double **A, int *P, double **L, double **U, int n, int nworkers)
{
  double global_max = 0.00;
  int global_k_prime = 0;
  double local_max = 0.00;
  int local_k_prime = 0;
  int start_row = 0;
  int tmp_k = 0;
  double *atmp_k = new double[n];
  double *ltmp_k = new double[n];
  int tmp_k_prime = 0;
  double *atmp_k_prime = new double[n];
  double *ltmp_k_prime = new double[n];
  int k_copied = 0;
  int k_primed_copied = 0;

  #pragma omp parallel num_threads(nworkers) shared(A, P, L, U)
  {
    int tid = omp_get_thread_num();

    // each thread takes on i idx to initialize P, L, U
    #pragma omp parallel for schedule (static, STATIC_CHUNK)
    for (int i = tid; i < n; i += nworkers)
    {
      L[i] = (double *)numa_alloc_local(n * sizeof(double));
      U[i] = (double *)numa_alloc_local(n * sizeof(double));
      memset(L[i], 0, n * sizeof(double));
      memset(U[i], 0, n * sizeof(double));
      L[i][i] = 1;
      P[i] = i + 1;
    }
  }

  #pragma omp parallel num_threads(nworkers) shared(A, P, L, U, global_max, global_k_prime, tmp_k, atmp_k, ltmp_k, tmp_k_prime, atmp_k_prime, ltmp_k_prime, k_copied, k_primed_copied) private(start_row, local_max, local_k_prime) firstprivate(n, nworkers)
  {
    int tid = omp_get_thread_num();
    for (int k = 0; k < n; k++)
    {
      start_row = tid + k / nworkers * nworkers;
      if (k % nworkers <= tid) {
        start_row = tid + k / nworkers * nworkers;
      }
      else {
        start_row = tid + (k / nworkers + 1) * nworkers;
      }
      #pragma omp parallel for schedule (static, STATIC_CHUNK)
      for (int i = start_row; i < n; i += nworkers)
      {
        // printf("tid  %d    i: %d\n", tid, i);
        if (local_max < abs(A[i][k]))
        {
          local_max = abs(A[i][k]);
          local_k_prime = i;
        }
      }

      #pragma omp critical (update_globals)
      {
        // printf("k: %d    tid %d   local_max: %f   setting globals: global_max: %f    global_k_prime: %d\n", k, tid, local_max, global_max, global_k_prime);
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
        // printf("global_max: %f    global_k_prime: %d\n", global_max, global_k_prime);
        if (global_max == (double) 0)
        {
          printf("error: singular matrix\n");
          exit(0);
        }
      }

      // #pragma omp single
      // {
      //   int tmp = P[k];
      //   P[k] = P[global_k_prime];
      //   P[global_k_prime] = tmp;

      //   double *atmp = A[k];
      //   A[k] = A[global_k_prime];
      //   A[global_k_prime] = atmp;

      //   // swap l(k, 1:k-1) and l(k', 1:k-1)
      //   double *ltmp = new double[n];
      //   memcpy(ltmp, L[k], k * sizeof(double));
      //   memcpy(L[k], L[global_k_prime], k * sizeof(double));
      //   memcpy(L[global_k_prime], ltmp, k * sizeof(double));

      //   memcpy(U[k] + k, A[k] + k, (n - k) * sizeof(double));
      //   global_max = 0.00;
      //   global_k_prime = 0;
      // }

      if (tid == k % nworkers) {
        tmp_k = P[k];
        memcpy(atmp_k, A[k], n * sizeof(double));
        memcpy(ltmp_k, L[k], k * sizeof(double));
      }

      if (tid == global_k_prime % nworkers) {
        tmp_k_prime = P[global_k_prime];
        memcpy(atmp_k_prime, A[global_k_prime], n * sizeof(double));
        memcpy(ltmp_k_prime, L[global_k_prime], k * sizeof(double));
      }

      #pragma omp barrier
      // #pragma omp single
      // {
      //   printf("k: %d    global_k_prime: %d\n", k, global_k_prime);
      // }
      if (tid == global_k_prime % nworkers) {
        // printf("k_prime  %d   tid  %d  in   tmp_k: %d\n", global_k_prime, tid, tmp_k);
        P[global_k_prime] = tmp_k;
        memcpy(A[global_k_prime], atmp_k, n * sizeof(double));
        memcpy(L[global_k_prime], ltmp_k, k * sizeof(double));
        k_copied = 1;
      }

      if (tid == k % nworkers) {
        // printf("k  %d   tid  %d  in   tmp_k_prime: %d\n", k, tid, tmp_k_prime);
        P[k] = tmp_k_prime;
        memcpy(A[k], atmp_k_prime, n * sizeof(double));
        memcpy(L[k], ltmp_k_prime, k * sizeof(double));
        memcpy(U[k] + k, A[k] + k, (n - k) * sizeof(double));
        k_primed_copied = 1;
      }

      if ((k + 1) % nworkers <= tid) {
        start_row = tid + (k + 1) / nworkers * nworkers;
      }
      else {
        start_row = tid + ((k + 1) / nworkers + 1) * nworkers;
      }
      #pragma omp barrier
      // #pragma omp single
      // {
      //   printf("k_copied: %d    k_primed_copied: %d\n", k_copied, k_primed_copied);
      // }
      if (k_copied == 1 && k_primed_copied == 1) {
        #pragma omp parallel for schedule (static)
        for (int i = start_row; i < n; i += nworkers)
        {
          // printf("tid  %d    i: %d\n", tid, i);
          L[i][k] = A[i][k] / U[k][k];
          #pragma omp parallel for schedule (dynamic)
          for (int j = k + 1; j < n; j++) {
            A[i][j] = A[i][j] - L[i][k] * U[k][j];
          }
        }
      }
      #pragma omp barrier
      #pragma omp single
      {
        global_max = 0.00;
        global_k_prime = 0;
        k_copied = 0;
        k_primed_copied = 0;
      }
    }
  }
}

void verification(double **A, int *P, double **L, double **U, int n)
{
  double diff = 0.00;
  #pragma omp parallel for schedule (static) shared(diff)
  for (int i = 0; i < n; i++)
  {
    double col_diff = 0.00;
    #pragma omp parallel for schedule (static)
    for (int j = 0; j < n; j++)
    {
      double tmp = A[P[i] - 1][j];
      #pragma omp parallel for schedule (static)
      for (int k = 0; k < n; k++)
      {
        tmp = tmp - L[i][k] * U[k][j];
      }
      col_diff = col_diff + tmp * tmp;
    }
    diff = diff + sqrt(col_diff);
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

  double **A;
  double **A1;
  int *P;
  double **L;
  double **U;
  // allocate memory for matrix A
  // malloc
  A = (double **)malloc(matrix_size * sizeof(double *));
  A1 = (double **)malloc(matrix_size * sizeof(double *));
  P = (int *)malloc(matrix_size * sizeof(int));
  L = (double **)malloc(matrix_size * sizeof(double *));
  U = (double **)malloc(matrix_size * sizeof(double *));

  random_number_generate(A, A1, matrix_size, nworkers);

  auto start = std::chrono::steady_clock::now();

  ludecomposition(A1, P, L, U, matrix_size, nworkers);

  auto end = std::chrono::steady_clock::now();

  auto diff = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  printf("elapsed time: %ds\n", diff);

  verification(A, P, L, U, matrix_size);

  // #pragma omp parallel for schedule (static)
  // for (int i = 0; i < matrix_size; i++)
  // {
  //   if (A[i] != NULL)
  //   {
  //     free(A[i]);
  //   }
  //   if (A1[i] != NULL)
  //   {
  //     free(A1[i]);
  //   }
  //   if (L[i] != NULL) {
  //     free(L[i]);
  //   }
  //   if (U[i] != NULL) {
  //     free(U[i]);
  //   }
  // }

  #pragma omp parallel num_threads(nworkers) shared(A, P, L, U) firstprivate(matrix_size, nworkers)
  {
    int tid = omp_get_thread_num();
    #pragma omp parallel for schedule (static)
    for (int i = tid; i < matrix_size; i += nworkers) {
      if (A[i] != NULL)
      {
        numa_free(A[i], matrix_size * sizeof(double));
      }
      if (A1[i] != NULL)
      {
        numa_free(A1[i], matrix_size * sizeof(double));
      } 
      if (L[i] != NULL) {
        numa_free(L[i], matrix_size * sizeof(double));
      }
      if (U[i] != NULL) {
        numa_free(U[i], matrix_size * sizeof(double));
      }
    }
  }
  free(A);
  free(A1);
  free(P);
  free(L);
  free(U);
  return 0;
}
