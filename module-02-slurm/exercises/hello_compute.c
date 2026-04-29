/*
 * hello_compute.c -- Report compute node details
 *
 * Compile:  gcc -o hello_compute hello_compute.c
 * Run:      sbatch submit_hello.sh
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void print_line_matching(const char *filename, const char *prefix) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return;

    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, prefix)) {
            /* Trim leading whitespace */
            char *p = line;
            while (*p == ' ' || *p == '\t') p++;
            printf("  %s", p);
        }
    }
    fclose(fp);
}

int main(void) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("=============================================\n");
    printf("  Hello from compute node: %s\n", hostname);
    printf("=============================================\n\n");

    /* CPU info */
    printf("--- CPU ---\n");
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (fp) {
        int cores = 0;
        int model_printed = 0;
        char line[512];
        while (fgets(line, sizeof(line), fp)) {
            if (strstr(line, "processor"))
                cores++;
            if (!model_printed && strstr(line, "model name")) {
                char *p = line;
                while (*p == ' ' || *p == '\t') p++;
                printf("  %s", p);
                model_printed = 1;
            }
        }
        fclose(fp);
        printf("  CPU cores visible: %d\n", cores);
    }

    /* Memory info */
    printf("\n--- Memory ---\n");
    print_line_matching("/proc/meminfo", "MemTotal");
    print_line_matching("/proc/meminfo", "MemAvailable");

    /* GPU check */
    printf("\n--- GPU ---\n");
    fflush(stdout);
    int ret = system("rocm-smi --showproductname 2>/dev/null");
    if (ret != 0) {
        printf("  rocm-smi not available or no GPU detected on this node.\n");
    }

    /* Environment variables set by Slurm */
    printf("\n--- Slurm Environment ---\n");
    const char *vars[] = {
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST", "SLURM_NTASKS", "SLURM_CPUS_ON_NODE",
        NULL
    };
    for (int i = 0; vars[i]; i++) {
        const char *val = getenv(vars[i]);
        printf("  %-25s = %s\n", vars[i], val ? val : "(not set)");
    }

    printf("\nDone.\n");
    return 0;
}
