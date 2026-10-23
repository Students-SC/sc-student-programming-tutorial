/*
 * offload_hello.c -- OpenMP GPU Offload Hello World
 *
 * Offloads a simple vector computation to the GPU and confirms whether it
 * actually ran on the device (GPU) or fell back to the host (CPU).
 *
 * Compile:  amdclang -O2 -fopenmp --offload-arch=gfx90a -o offload_hello offload_hello.c
 * Run:      ./offload_hello   (on a GPU compute node)
 */

#include <stdio.h>
#include <omp.h>

#define N 100000

int main(void) {
    double a[N];
    int ran_on_device = 0;

    printf("Number of GPU devices visible to OpenMP: %d\n", omp_get_num_devices());

    /* Offload the loop to the GPU. The map clauses copy 'a' back from the
     * device and copy the flag both ways so we can report where it ran. */
    #pragma omp target teams distribute parallel for \
            map(from:a[0:N]) map(tofrom:ran_on_device)
    for (int i = 0; i < N; i++) {
        a[i] = 2.0 * i;
        if (i == 0) {
            /* omp_is_initial_device() is 0 on the GPU, 1 on the host CPU */
            ran_on_device = !omp_is_initial_device();
        }
    }

    printf("a[0] = %.1f, a[N-1] = %.1f\n", a[0], a[N - 1]);

    if (ran_on_device)
        printf("The target region ran on the GPU device. \\o/\n");
    else
        printf("The target region ran on the host CPU (no GPU offload).\n");

    return 0;
}
