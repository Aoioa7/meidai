#include <stdio.h>
#include <math.h>
void main(){
int j,k,kk,n;
float lmd,alp,bet,pi,h,dt,x,t;
float u[100][100],ue[100][100];
FILE *fp;
lmd=1.0;
alp=0.0;
bet=0.0;
pi=3.14159;
n=9;
h=0.1;
dt=0.005;
kk=5;
for(j=0;j<n+2;j++)
for(k=0;k<kk+1;k++)
u[j][k]=0.0;
for(j=1;j<n+1;j++){
x=h*(float)j;
u[j][0]=sin(pi*x);
}
for(k=0;k<kk+1;k++){
u[0][k+1]=alp;
u[n+1][k+1]=bet;
for(j=1;j<n+1;j++)
u[j][k+1]=u[j][k]+lmd*dt/h/h*(u[j-1][k]-2.0*u[j][k]+u[j+1][k]);
}
for(k=0;k<kk+1;k++){
t=k*dt;
for(j=0;j<n+2;j++){
x=h*(float)j;
ue[j][k]=exp(-pi*pi*t)*sin(pi*x);
}
}
fp=fopen("result_600.dat","w");
fprintf(fp," k ");
for(k=0;k<kk+1;k++)
fprintf(fp,"%2d ",k);
fprintf(fp,"解析解(k=4)¥n");
for(j=0;j<n+2;j++){
fprintf(fp,"j=%2d ",j);
for(k=0;k<kk+1;k++){
fprintf(fp,"%8.4f",u[j][k]);
}
fprintf(fp,"%8.4f¥n",ue[j][kk]);
}
fclose(fp);
for(j=0;j<n+2;j++){
printf(" j= %2d",j);
for(k=0;k<kk+1;k++)
printf("%8.4f",u[j][k]);
printf("¥n");
}
}
