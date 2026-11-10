/*
 * pi_serial.c -- Estimate pi using numerical integration (serial)
 *
 * Method: pi = integral from 0 to 1 of 4/(1+x^2) dx
 * We approximate this with a Riemann sum over N steps.
 *
 * Compile:  gcc -O2 -o pi_serial pi_serial.c -lm
 *           (-O2 asks the compiler to optimize the code for speed;
 *            -lm links the math library, which provides fabs())
 * Run:      ./pi_serial
 */

/* Header files declare the library functions used below. */
#include <stdio.h>   /* printf()                         */
#include <math.h>    /* fabs() and the constant M_PI     */
#include <time.h>    /* clock_gettime(), for timing      */

/* The number of rectangles in the Riemann sum. More steps give a more
 * accurate answer and take longer. The L makes it a long integer. */
#define NUM_STEPS 1000000000L

/* Every C program starts running at main(). */
int main(void) {
    /* Each step is a slice of width 1/NUM_STEPS on the interval [0, 1].
     * "double" is a 64-bit floating-point number. */
    double step = 1.0 / (double)NUM_STEPS;
    double sum = 0.0;

    /* Record the start time. */
    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* The main loop: for each slice, evaluate 4/(1+x^2) at the slice's
     * midpoint x and add it to the running sum. This is the loop you
     * will parallelize in Exercise 3. */
    for (long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    /* Each slice is a rectangle of height f(x) and width step, so the area
     * under the curve -- our estimate of pi -- is the sum times the width. */
    double pi = sum * step;

    /* Record the end time; elapsed = end - start, in seconds. */
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    /* Print the results. In printf, %.15f prints a number with 15 digits
     * after the decimal point, %.2e prints it in scientific notation, and
     * %ld prints a long integer. */
    printf("Computed pi = %.15f\n", pi);
    printf("Reference   = %.15f\n", M_PI);
    printf("Error       = %.2e\n", fabs(pi - M_PI));
    printf("Steps       = %ld\n", NUM_STEPS);
    printf("Time        = %.3f seconds\n", elapsed);

    /* Returning 0 from main() tells the shell the program succeeded. */
    return 0;
}
