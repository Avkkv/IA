/*
 * 02 - Multiplicacao de matrizes em C, em quatro versoes.
 * A multiplicacao de matrizes (GEMM) e onde um modelo de linguagem gasta quase
 * todo o tempo. Aqui voce ve, na pratica, por que a ordem dos lacos, o cache,
 * o SIMD e as threads mudam o desempenho em ordens de grandeza.
 *
 * Todas calculam C = A * B para matrizes quadradas n x n, float32, row-major.
 *
 * Compilar (o bench.py faz isso sozinho):
 *   gcc -O3 -march=native -fopenmp -shared -fPIC matmul.c -o libmatmul.so
 */
#include <string.h>

/* v1: ingenua (i,j,k). B e lida "pulando" n floats por vez: ruim para o cache. */
void matmul_ingenua(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++)
                s += A[i * n + k] * B[k * n + j];
            C[i * n + j] = s;
        }
}

/* v2: troca a ordem para (i,k,j). O laco interno percorre B e C em sequencia,
 * o que e amigavel ao cache e permite ao compilador vetorizar (SIMD/AVX). */
void matmul_ikj(const float *A, const float *B, float *C, int n) {
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

/* v3: blocagem (tiling). Trabalha em blocos bs x bs que cabem no cache L1/L2,
 * reaproveitando cada dado carregado varias vezes antes de ele sair do cache. */
void matmul_blocada(const float *A, const float *B, float *C, int n, int bs) {
    memset(C, 0, (size_t)n * n * sizeof(float));
    for (int ii = 0; ii < n; ii += bs)
        for (int kk = 0; kk < n; kk += bs)
            for (int jj = 0; jj < n; jj += bs) {
                int imax = ii + bs < n ? ii + bs : n;
                int kmax = kk + bs < n ? kk + bs : n;
                int jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        float a = A[i * n + k];
                        for (int j = jj; j < jmax; j++)
                            C[i * n + j] += a * B[k * n + j];
                    }
            }
}

/* v4: blocada + varias threads (OpenMP). Cada thread cuida de faixas de linhas
 * de C, entao nao ha disputa de escrita entre elas. */
void matmul_paralela(const float *A, const float *B, float *C, int n, int bs) {
    memset(C, 0, (size_t)n * n * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += bs)
        for (int kk = 0; kk < n; kk += bs)
            for (int jj = 0; jj < n; jj += bs) {
                int imax = ii + bs < n ? ii + bs : n;
                int kmax = kk + bs < n ? kk + bs : n;
                int jmax = jj + bs < n ? jj + bs : n;
                for (int i = ii; i < imax; i++)
                    for (int k = kk; k < kmax; k++) {
                        float a = A[i * n + k];
                        for (int j = jj; j < jmax; j++)
                            C[i * n + j] += a * B[k * n + j];
                    }
            }
}
