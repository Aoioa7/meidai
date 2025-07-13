#include <stdio.h>
#include <math.h>

int main(){
    int i, j, n;
    double p, q, f, alp, bet, h;
    double a[100], b[100], c[100], d[100], g[100];
    double v[100], u[100], m[100];

    // ——— パラメータ設定 ———
    p   = 1.0;     // 定数係数 p(x)=1
    q   = 16.0;    // u の項の係数
    f   = 0.0;     // ← f は定数ではなく、後で x_j を使って構成する
    alp = 0.0;     // u(0)=0
    bet = 0.0;     // u(1)=0

    n   = 9;                  // 内部分割数
    h   = 1.0 / (n + 1);       // 格子幅

    // ——— tridiagonal 行列の組み立て ———
    // 対角成分 a[j] = 2*p + q*h^2
    for(j = 0; j < n; j++){
        a[j] = 2.0*p + q*h*h;
    }
    // 上下対角
    for(j = 0; j < n-1; j++){
        b[j]   = -p;   // 上
        c[j+1] = -p;   // 下
    }

    // ——— 右辺ベクトル g の構築 ———
    // 本来 f(x)=x なので、各格子点 x_j=(j+1)*h を代入
    for(j = 0; j < n; j++){
        double xj = (j + 1) * h;
        g[j]      = h*h * xj;
    }
    // 境界条件の寄与（今回は alp=bet=0 なので変化なし）
    g[0]     += p * alp;
    g[n-1]   += p * bet;

    // ——— Thomas アルゴリズム（前進消去） ———
    d[0] = a[0];
    for(i = 1; i < n; i++){
        m[i]   = c[i] / d[i-1];
        d[i]   = a[i] - m[i] * b[i-1];
    }

    // ——— Thomas アルゴリズム（前進代入） ———
    v[0] = g[0];
    for(i = 1; i < n; i++){
        v[i] = g[i] - m[i] * v[i-1];
    }

    // ——— Thomas アルゴリズム（後退代入） ———
    u[n-1] = v[n-1] / d[n-1];
    for(i = n-2; i >= 0; i--){
        u[i] = (v[i] - b[i] * u[i+1]) / d[i];
    }

    // ——— 結果表示 ———
    for(i = 0; i < n; i++){
        printf(" i=%2d  u= %12.8f\n", i+1, u[i]);
    }

    return 0;
}
