#include <stdio.h>
#include <math.h>

// マクロ化
#define N   9              
#define H   0.1            
#define DT  0.005          
#define KK  60             
#define PI  3.141592653589793

int main(void) {
    // doubleにして精度を上げる
    double u[N+2][KK+1];
    double ue[N+2][KK+1];
    double lmd = 1.0, alp = 0.0, bet = 0.0;
    FILE *fp;
    int j, k;

    for (j = 0; j < N+2; j++) {
        for (k = 0; k <= KK; k++) {
            u[j][k] = 0.0;
        }
    }

    for (j = 1; j <= N; j++) {
        double x = j * H;
        u[j][0] = sin(PI * x);
    }

    for (k = 0; k < KK; k++) {
        u[0][k+1]   = alp;
        u[N+1][k+1] = bet;
        for (j = 1; j <= N; j++) {
            u[j][k+1] = u[j][k]
                      + lmd * DT / (H * H)
                      * (u[j-1][k] - 2.0 * u[j][k] + u[j+1][k]);
        }
    }

    for (k = 0; k <= KK; k++) {
        double t = k * DT;
        for (j = 0; j < N+2; j++) {
            double x = j * H;
            ue[j][k] = exp(-PI * PI * t) * sin(PI * x);
        }
    }

    int idx[] = {0, 10, 20, 30, 40, 50, 60};
    int nidx = sizeof(idx) / sizeof(idx[0]);

    fp = fopen("result_600.dat", "w");
    fprintf(fp, " k ");
    for (int i = 0; i < nidx; i++) {
        fprintf(fp, "%2d ", idx[i]);
    }
    fprintf(fp, "解析解(k=%d)\n", idx[nidx-1]);

    for (j = 0; j < N+2; j++) {
        fprintf(fp, "j=%2d ", j);
        for (int i = 0; i < nidx; i++) {
            fprintf(fp, "%8.4f", u[j][ idx[i] ]);
        }
        fprintf(fp, "%8.4f\n", ue[j][ idx[nidx-1] ]);
    }
    fclose(fp);

    for (j = 0; j < N+2; j++) {
        printf(" j= %2d", j);
        for (int i = 0; i < nidx; i++) {
            printf("%8.4f", u[j][ idx[i] ]);
        }
        printf("\n");
    }

    return 0;
}
