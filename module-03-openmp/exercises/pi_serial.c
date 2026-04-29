/*
 * pi_serial.c -- Estimate pi using numerical integration (serial)
 *
 * Method: pi = integral from 0 to 1 of 4/(1+x^2) dx
 * We approximate this with a Riemann sum over N steps.
 *
 * Compile:  gcc -O2 -o pi_serial pi_serial.c -lm
 * Run:      ./pi_serial
 */

#include <stdio.h>
#include <math.h>
#include <time.h>

#define NUM_STEPS 100000000L

int main(void) {
    double step = 1.0 / (double)NUM_STEPS;
    double sum = 0.0;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = sum * step;

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("Computed pi = %.15f\n", pi);
    printf("Reference   = %.15f\n", M_PI);
    printf("Error       = %.2e\n", fabs(pi - M_PI));
    printf("Steps       = %ld\n", NUM_STEPS);
    printf("Time        = %.3f seconds\n", elapsed);

    return 0;
}
