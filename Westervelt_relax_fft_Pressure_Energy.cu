#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cufft.h> 
// !!! don't forget to link the CUFFT library by edit to the end: nvcc ... -lcufft


// medium properties
// water
__constant__ double rho; // tissue density, [kg/m^3]
__constant__ double c_inf; // speed of sound
__constant__ double delta; // diffusivity
__constant__ double beta; // nonlinearity
__constant__ double abs_water; // absorbtion coefficient

// tissue (muscle, pork)
__constant__ double rho_tissue; // tissue density, [kg/m^3]
__constant__ double c_inf_tissue; // speed of sound
__constant__ double beta_tissue; // nonlinearity
__constant__ double abs_tissue; // absorbtion coefficient

// relaxation parameters
__constant__ double delta_tissue; // diffusivity
__constant__ double c1; // speed in relaxation process
__constant__ double c2; // speed in relaxation process
__constant__ double tau1; // relaxation time
__constant__ double tau2; // relaxation time


// transducer parameters
__constant__ double fr; // ultrasound frequency
__constant__ double d; // focal length [m]
__constant__ double alpha; // aperture angle
__constant__ double alpha_hole; // aperture angle of hole in the center of transducer

__constant__ double p0; // on-source pressure

__constant__ double water_tissue; // 0.12; 0.115 


// mesh parameters
__constant__ double xy_min;
__constant__ double dxy;
__constant__ double z_min;
__constant__ double dz;
__constant__ double dt; // time step for Westervelt equation









__global__ void boundary_condition_source(double* P, const double t, const int Nxy, const int Nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // columns // radial axis
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis
    
    int Nxy_2 = Nxy * Nxy;
    int idx = iz * Nxy_2 + ix + iy * Nxy; // global index
    
    // source pressure (min at focal axis) (water)
    double X = xy_min + ix * dxy;
    double Y = xy_min + iy * dxy;
    double r = sqrt(X*X + Y*Y); // in 2D: r = abs(X); // in 3D: r = sqrt(X*X + Y*Y)
    if (iz == 2 && ix < Nxy && iy < Nxy) // iz == 2 => Z = 0
    {
        P[idx] = 0.0;
        if (r <= d*tan(alpha) && r >= d*tan(alpha_hole))
        {
            double var = sqrt(1.0 + pow(r/d, 2));
            double tt = t + (d*(var - 1.0) / c_inf); 
            P[idx] = p0 * sin(2*M_PI*fr*tt) / var;
        }
    }
}


__global__ void boundary_condition_space6(double* P, double* Pm1, const double t, const int Nxy, const int Nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // columns // radial axis
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis
    
    int Nxy_2 = Nxy * Nxy;
    int idx = iz * Nxy_2 + ix + iy * Nxy; // global index
    
    // source pressure (min at focal axis) --> in separated function
    
    // min at Z focal axis (water)
    double cdt_dz = c_inf * dt / dz;
    if (iz >= 0 && iz <= 1 && ix < Nxy && iy < Nxy)
    {
        int zp = (iz + 1) * Nxy_2 + ix + iy * Nxy;
        P[idx] = Pm1[zp] + ((P[zp] - Pm1[idx]) * (cdt_dz - 1.0) / (cdt_dz + 1.0));
    }
    
    // max at focal axis (tissue)
    if (iz <= Nz-1 && iz >= Nz-3 && ix < Nxy && iy < Nxy)
    {
        int zm = (iz - 1) * Nxy_2 + ix + iy * Nxy;
        double cdt_dz_tissue = c_inf_tissue * dt / dz;
        P[idx] = Pm1[zm] + ((P[zm] - Pm1[idx]) * (cdt_dz_tissue - 1.0) / (cdt_dz_tissue + 1.0));
    }
    
    double cdt_dxy = 0.0;
    if (z_min + iz*dz < water_tissue) // water
        cdt_dxy = c_inf * dt / dxy;
    else // tissue
        cdt_dxy = c_inf_tissue * dt / dxy;
        
    // max at X radial axis
    if (ix <= Nxy-1 && ix >= Nxy-3 && iz < Nz && iy < Nxy)
    {
        int xm = iz * Nxy_2 + (ix - 1) + iy * Nxy;
        P[idx] = Pm1[xm] + ((P[xm] - Pm1[idx]) * (cdt_dxy - 1.0) / (cdt_dxy + 1.0));
    }
    
    // max at Y radial axis
    if (ix < Nxy && iz < Nz && iy <= Nxy-1 && iy >= Nxy-3)
    {
        int ym = iz * Nxy_2 + ix + (iy - 1) * Nxy;
        P[idx] = Pm1[ym] + ((P[ym] - Pm1[idx]) * (cdt_dxy - 1.0) / (cdt_dxy + 1.0));
    }
    
}



__global__ void Westervelt_space6(double* Pnext, double* P, double* Pm1, double* Pm2, double* Pm3, double* Pm4, double* R1, double* R1m1, double* R1m2, double* R2, double* R2m1, double* R2m2, const int Nxy, const int Nz)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // columns // radial axis
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis
    
    
    // for total inside points
    if (iz >= 3 && iz <= Nz-4 && ix >= 0 && ix <= Nxy-4 && iy >= 0 && iy <= Nxy-4)
	{
        double c2dt2_dxy2 = 0.0; // c^2 * dt^2 / dxy^2
        double c2dt2_dz2 = 0.0; // c^2 * dt^2 / dz^2
        double delta_2dtc2 = 0.0; // delta / (2*dt * c^2)
        double beta2_rhoc2 = 0.0; // 2*beta / (rho * c^2)
        
        int Nxy_2 = Nxy * Nxy;
        int idx = iz * Nxy_2 + ix + iy * Nxy; // global index
        
        int xp1 = iz * Nxy_2 + (ix + 1) + iy * Nxy;
        int xp2 = iz * Nxy_2 + (ix + 2) + iy * Nxy;
        int xp3 = iz * Nxy_2 + (ix + 3) + iy * Nxy;
        int xm1 = iz * Nxy_2 + (ix - 1) + iy * Nxy;
        int xm2 = iz * Nxy_2 + (ix - 2) + iy * Nxy;
        int xm3 = iz * Nxy_2 + (ix - 3) + iy * Nxy;
        int yp1 = iz * Nxy_2 + ix + (iy + 1) * Nxy;
        int yp2 = iz * Nxy_2 + ix + (iy + 2) * Nxy;
        int yp3 = iz * Nxy_2 + ix + (iy + 3) * Nxy;
        int ym1 = iz * Nxy_2 + ix + (iy - 1) * Nxy;
        int ym2 = iz * Nxy_2 + ix + (iy - 2) * Nxy;
        int ym3 = iz * Nxy_2 + ix + (iy - 3) * Nxy;
        int zp1 = (iz + 1) * Nxy_2 + ix + iy * Nxy;
        int zp2 = (iz + 2) * Nxy_2 + ix + iy * Nxy;
        int zp3 = (iz + 3) * Nxy_2 + ix + iy * Nxy;
        int zm1 = (iz - 1) * Nxy_2 + ix + iy * Nxy;
        int zm2 = (iz - 2) * Nxy_2 + ix + iy * Nxy;
        int zm3 = (iz - 3) * Nxy_2 + ix + iy * Nxy;
        
        if (ix - 1 == -1) // => ix == 0
            xm1 = iz * Nxy_2 + (ix + 1) + iy * Nxy;
        if (ix - 2 == -1) // => ix == 1
            xm2 = iz * Nxy_2 + ix + iy * Nxy;
        if (ix - 2 == -2) // => ix == 0
            xm2 = iz * Nxy_2 + (ix + 2) + iy * Nxy;
        if (ix - 3 == -1) // => ix == 2
            xm3 = iz * Nxy_2 + (ix - 1) + iy * Nxy;
        if (ix - 3 == -2) // => ix == 1
            xm3 = iz * Nxy_2 + (ix + 1) + iy * Nxy;
        if (ix - 3 == -3) // => ix == 0
            xm3 = iz * Nxy_2 + (ix + 3) + iy * Nxy;
            
        if (iy - 1 == -1) // => iy == 0
            ym1 = iz * Nxy_2 + ix + (iy + 1) * Nxy;
        if (iy - 2 == -1) // => iy == 1
            ym2 = iz * Nxy_2 + ix + iy * Nxy;
        if (iy - 2 == -2) // => iy == 0
            ym2 = iz * Nxy_2 + ix + (iy + 2) * Nxy;
        if (iy - 3 == -1) // => iy == 2
            ym3 = iz * Nxy_2 + ix + (iy - 1) * Nxy;
        if (iy - 3 == -2) // => iy == 1
            ym3 = iz * Nxy_2 + ix + (iy + 1) * Nxy;
        if (iy - 3 == -3) // => iy == 0
            ym3 = iz * Nxy_2 + ix + (iy + 3) * Nxy;
        
        
        double Dxx = (-490.0*P[idx] + 270.0*(P[xp1] + P[xm1]) - 27.0*(P[xp2] + P[xm2]) + 2.0*(P[xp3] + P[xm3])) / 180.0;
        double Dyy = (-490.0*P[idx] + 270.0*(P[yp1] + P[ym1]) - 27.0*(P[yp2] + P[ym2]) + 2.0*(P[yp3] + P[ym3])) / 180.0;
        double Dzz = (-490.0*P[idx] + 270.0*(P[zp1] + P[zm1]) - 27.0*(P[zp2] + P[zm2]) + 2.0*(P[zp3] + P[zm3])) / 180.0;
        
        double Dttt = 5.0*P[idx] - 18.0*Pm1[idx] + 24.0*Pm2[idx] - 14.0*Pm3[idx] + 3.0*Pm4[idx];
        double D2t2_p1 = pow(3.0*P[idx] - 4.0*Pm1[idx] + Pm2[idx], 2) / 4.0;
        double D2t2_p2 = P[idx] * (2.0*P[idx] - 5.0*Pm1[idx] + 4.0*Pm2[idx] - Pm3[idx]);
        
        double sum_R = 0.0; // sum of special variables for relaxtion = R1 + R2
        if (z_min + iz*dz < water_tissue) // water => without relaxation
        {
            c2dt2_dxy2 = pow(c_inf * dt / dxy, 2);
            c2dt2_dz2 = pow(c_inf * dt / dz, 2);
            delta_2dtc2 = delta / (2.0 * dt * pow(c_inf, 2));
            beta2_rhoc2 = 2.0 * beta / (rho * pow(c_inf, 2));
        }
        else // tissue => relaxation
        {
            c2dt2_dxy2 = pow(c_inf_tissue * dt / dxy, 2);
            c2dt2_dz2 = pow(c_inf_tissue * dt / dz, 2);
            delta_2dtc2 = delta_tissue / (2.0 * dt * pow(c_inf_tissue, 2));
            beta2_rhoc2 = 2.0 * beta_tissue / (rho_tissue * pow(c_inf_tissue, 2));
            
            double Dttt_2dt = Dttt / (2.0 * pow(dt, 3));
            R1[idx] = (2.0*dt * 2.0 * c1*tau1*Dttt_2dt / ((2.0*dt + 3.0*tau1)*pow(c_inf_tissue, 3))) + ((tau1*(4.0*R1m1[idx] - R1m2[idx])) / (2.0*dt + 3.0*tau1));
            R2[idx] = (2.0*dt * 2.0 * c2*tau2*Dttt_2dt / ((2.0*dt + 3.0*tau2)*pow(c_inf_tissue, 3))) + ((tau2*(4.0*R2m1[idx] - R2m2[idx])) / (2.0*dt + 3.0*tau2));
            
            sum_R = (R1[idx] + R2[idx]) * pow(c_inf_tissue * dt, 2);
        }
        
        
        Pnext[idx] = 2.0*P[idx] - Pm1[idx] + c2dt2_dxy2*(Dxx + Dyy) + c2dt2_dz2*Dzz + delta_2dtc2*Dttt + beta2_rhoc2*(D2t2_p1 + D2t2_p2) + sum_R;
    }
}





__global__ void acoustic_power_water(double* Q, double* Pnext, double* Pm1, const int Nxy, const int Nz, const bool last)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // columns // radial axis
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis
    
    double Z = z_min + iz*dz;
    if (ix >= 0 && ix < Nxy && iy >= 0 && iy < Nxy && iz >= 0 && Z < water_tissue) // water
    {
        int idx = iz*Nxy*Nxy + ix + iy * Nxy; // global index
        
        double dP_dt_1 = (Pnext[idx] - Pm1[idx]) / (2.0*dt); // n
        
        if (!last) // just calculate time average (dP/dt)^2
        {
            Q[idx] = Q[idx] + pow(dP_dt_1, 2)*dt;
        }
        else // final formula
        {
            Q[idx] = 2.0*abs_water*fr*Q[idx] / (pow(2.0*M_PI*fr, 2)*rho*c_inf); // since 1/T = fr, where T=period=1/f
        }
        
    }
}


__global__ void Double_to_Complex(double* R, cufftComplex* C, const int nx, const int Nxy, const int Nz_tissue, const int N_time)
{
    int i_time = threadIdx.x + blockDim.x * blockIdx.x; // columns // time
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis (tissue)
    
    if (i_time < N_time && iy < Nxy && iz < Nz_tissue)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            int i_xyz = iz * Nxy*nx + ix + iy*nx;
            int idx = i_time + i_xyz * N_time;
            C[idx].x = R[idx];
            C[idx].y = 0.0;
        }
    }
}


__global__ void Abs_Complex(cufftComplex* C, double* A, const int nx, const int Nxy, const int Nz_tissue, const int N_time)
{
    int i_time = threadIdx.x + blockDim.x * blockIdx.x; // columns // time
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis (tissue)
    
    if (i_time < N_time && iy < Nxy && iz < Nz_tissue)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            int i_xyz = iz * Nxy*nx + ix + iy*nx;
            int idx = i_time + i_xyz * N_time;
            A[idx] = sqrt(pow(C[idx].x, 2) + pow(C[idx].y, 2)) / N_time;
        }
    }
}



__global__ void copy_data(double* P_next, double* Pxyz_time, const int i_nx, const int nx, const int Nxy, const int Nz_tissue, const int iz_tissue, const int i_time, const int N_time)
{
    // common for P_next and Pxy_time
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // columns // radial axis
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis
    
    if (ix < nx && iy < Nxy && iz < Nz_tissue)
    {
        int idx = ((iz + iz_tissue) * Nxy*Nxy) + (ix + i_nx*nx) + iy * Nxy; // global index for P_next
        
        int i_xyz = iz * Nxy*nx + ix + iy*nx;
        int idx_time = i_time + i_xyz * N_time; // global index for Pxyz_time
        Pxyz_time[idx_time] = P_next[idx];
    }   
}


__global__ void total_energy_Nharmonics_xyz(double* Q, double* Pxyz_abs, const int i_nx, const int nx, const int Nxy, const int Nz_tissue, const int iz_tissue, 
const int N_time, const int periods, const int N_harm)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // columns // radial axis
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows // radial axis
    int iz = threadIdx.z + blockDim.z * blockIdx.z; // matrices // focal axis
    
    int idx = ((iz + iz_tissue) * Nxy*Nxy) + (ix + i_nx*nx) + iy * Nxy; // global index
    
    if (ix < nx && iy < Nxy && iz < Nz_tissue) // tissue
    {
        double In = 0;
        Q[idx] = 0;
        int i_xyz = iz * Nxy*nx + ix + iy*nx;
        int idx_time = i_xyz * N_time;
        for (int n = 1; n <= N_harm; n++)
        {
            In = 2.0 * pow(Pxyz_abs[idx_time + n*periods], 2) / (rho_tissue * c_inf_tissue);
            Q[idx] = Q[idx] + (2.0 * (7.2 * n*fr/pow(10.0, 6)) * In);
        }
    }
}




int main()
{
	/* SET DEVICE*/
	cudaSetDevice(0);
    
    // transducer parameters [big transducer]
    double h_fr = 1.253 * pow(10.0, 6); // ultrasound frequency
    double h_d = 0.15; // focal length [m]
    double a = 0.151 / 2.0; // radius [m]
    double h = h_d - sqrt((h_d*h_d) - (a*a)); // transducer depth
    double a_hole = 0.041 / 2.0; // radius of center hole
    
    double h_alpha = asin(a / h_d); // aperture angle
    double h_alpha_hole = asin(a_hole / h_d); // aperture angle of hole in the center of transducer
    
    // water parameters
    double rho_water = 1000.0; // density
    double c_inf_water = 1500.0; // speed of sound
    double c_t_water = 4200.0; // 4000.0; // specific heat capacity [J/(kg*K)]
    double K_t_water = 0.6; // thermal conductivity [W/(m*K)]
    double k_water = K_t_water / (rho_water * c_t_water);
    double h_abs_water = 0.026 * pow(h_fr / pow(10.0, 6), 2); // acoustic absorption coefficient [Np/m]
    double delta_water = (2.0 * h_abs_water * pow(c_inf_water, 3)) / pow(2.0*M_PI*h_fr, 2); // about 4.6257 * pow(10.0, -6); // acoustic diffusivity
    double beta_water = 3.5; // 3.5; // nonlinear coefficient
    
    // tissue (muscle) parameters [Maxim_2014]
    double h_rho_tissue = 1055.0; // density
    double h_c_inf_tissue = 1600.0; // 1600.0; // 1550.0; // speed of sound
    double c_t_tissue = 3700.0; // 3200.0; // tissue specific heat [J/(kg*K)]
	double K_t_tissue = 0.51; // 0.49; // tissue thermal conductivity [W/(m*K)]
	double k_tissue = K_t_tissue / (h_rho_tissue * c_t_tissue); 
    double h_beta_tissue = 4.5; // 4.5; // nonlinear coefficient
    double h_abs_tissue = 7.2 * h_fr / pow(10.0, 6); // acoustic absorption coefficient [Np/m]
    double att_tissue = 9.0 * h_fr / pow(10.0, 6); // acoustic attenuation coefficient [Np/m]
    
    
    // estimated parameters for 2 relaxation processes
    // att = 9.0*fr, c0 = 1600
    /**/
    double h_delta_tissue = 4.2318 * pow(10.0, -5); // acoustic diffusivity
    double h_c1 = 4.03536; // small speed of sound, m/s
    double h_c2 = 6.14150; // small speed of sound, m/s
    double h_tau1 = 1.6178 * pow(10.0, -8); // relaxation time, s
    double h_tau2 = 1.1494 * pow(10.0, -7); // relaxation time, s
    /**/
    
    double eps = 0.89; // transducer efficiency
    double E = 100.0; // 150.0; // energy for transducer [W]
    double u0_water = sqrt(2.0*E*eps / (rho_water*c_inf_water*2.0*M_PI*h_d*h)); // particals velocity
        
    double h_p0 = rho_water*c_inf_water*u0_water; // on-source pressure [Pa] (see [Hamilton_1992])
    
    double h_water_tissue = 0.12; // 0.12 // 0.115 where water ends and tissue starts
    
    
    // mesh parameters
    double lambda = c_inf_water / h_fr; // wavelength [m]
    
    double h_xy_min = 0.0, xy_max = 0.085; // [m]
    double h_dxy = 0.0002; // 0.2 * lambda; // 0.0002; // 0.00015
    int Nxy = 1 + round((xy_max - h_xy_min) / h_dxy);
    
    double z_max = 0.2, h_z_min = -0.0002; // focal axis // h_z_min = 0.0
    double h_dz = 0.0001; // 0.1 * lambda; // 0.0001; // 0.0002
    int Nz = 1 + round((z_max - h_z_min) / h_dz);
        
    double h_dt = 0.01 / h_fr; // pow(10.0, -8);
    double t_max = 165.0 / h_fr; // 100.0 / h_fr; // # periods
    
    
    printf("\n xy = [%.3f; %.3f]m; Nxy = %d \n", h_xy_min, xy_max, Nxy);
	printf("\n z = [%.4f; %.3f]m; Nz = %d \n", h_z_min, z_max, Nz);
    printf("\n dxy = %.4f m; dz = %.4f m \n", h_dxy, h_dz);
	printf("\n dt = %.10f; t_max = %f \n", h_dt, t_max);
    printf("\n Water ends at Z = %.3f; \n", h_water_tissue);
    
        

	// create the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start the timer
	cudaEventRecord(start, 0);
    

	// copy from host to global device memory    
    cudaMemcpyToSymbol(rho, &rho_water, sizeof(double));
    cudaMemcpyToSymbol(c_inf, &c_inf_water, sizeof(double));
    cudaMemcpyToSymbol(delta, &delta_water, sizeof(double));
    cudaMemcpyToSymbol(beta, &beta_water, sizeof(double));
    cudaMemcpyToSymbol(abs_water, &h_abs_water, sizeof(double));
    
    cudaMemcpyToSymbol(rho_tissue, &h_rho_tissue, sizeof(double));
    cudaMemcpyToSymbol(c_inf_tissue, &h_c_inf_tissue, sizeof(double));
    cudaMemcpyToSymbol(beta_tissue, &h_beta_tissue, sizeof(double));
    cudaMemcpyToSymbol(abs_tissue, &h_abs_tissue, sizeof(double));
    
    cudaMemcpyToSymbol(delta_tissue, &h_delta_tissue, sizeof(double));
    cudaMemcpyToSymbol(c1, &h_c1, sizeof(double));
    cudaMemcpyToSymbol(c2, &h_c2, sizeof(double));
    cudaMemcpyToSymbol(tau1, &h_tau1, sizeof(double));
    cudaMemcpyToSymbol(tau2, &h_tau2, sizeof(double));
    
    cudaMemcpyToSymbol(fr, &h_fr, sizeof(double));
    cudaMemcpyToSymbol(d, &h_d, sizeof(double));
    cudaMemcpyToSymbol(alpha, &h_alpha, sizeof(double));
    cudaMemcpyToSymbol(alpha_hole, &h_alpha_hole, sizeof(double));
    cudaMemcpyToSymbol(p0, &h_p0, sizeof(double));
    
    cudaMemcpyToSymbol(water_tissue, &h_water_tissue, sizeof(double)); 

    cudaMemcpyToSymbol(xy_min, &h_xy_min, sizeof(double));
    cudaMemcpyToSymbol(dxy, &h_dxy, sizeof(double));
    cudaMemcpyToSymbol(z_min, &h_z_min, sizeof(double)); 
    cudaMemcpyToSymbol(dz, &h_dz, sizeof(double));
    cudaMemcpyToSymbol(dt, &h_dt, sizeof(double));
    

	int N_total = Nxy*Nxy * Nz;
	double* h_P_next = new double[N_total];
    double* h_P = new double[N_total];
    double* h_P_m1 = new double[N_total];
    double* h_P_m2 = new double[N_total];
    double* h_P_m3 = new double[N_total];
    double* h_P_m4 = new double[N_total];
    
    double* h_R1 = new double[N_total];
    double* h_R1m1 = new double[N_total];
    double* h_R1m2 = new double[N_total];
    double* h_R2 = new double[N_total];
    double* h_R2m1 = new double[N_total];
    double* h_R2m2 = new double[N_total];


	// initial conditions
    int idx = 0;
	for (idx = 0; idx < N_total; idx++) 
	{
		h_P_next[idx] = 0.0;
        h_P[idx] = 0.0;
        h_P_m1[idx] = 0.0;
        h_P_m2[idx] = 0.0;
        h_P_m3[idx] = 0.0;
        h_P_m4[idx] = 0.0;
        h_R1[idx] = 0.0;
        h_R1m1[idx] = 0.0;
        h_R1m2[idx] = 0.0;
        h_R2[idx] = 0.0;
        h_R2m1[idx] = 0.0;
        h_R2m2[idx] = 0.0;
	}
    
	double* d_P_next;
    double* d_P;
    double* d_P_m1;
    double* d_P_m2;
    double* d_P_m3;
    double* d_P_m4;
	cudaMalloc((void**)&d_P_next, N_total * sizeof(double));
	cudaMalloc((void**)&d_P, N_total * sizeof(double));
    cudaMalloc((void**)&d_P_m1, N_total * sizeof(double));
    cudaMalloc((void**)&d_P_m2, N_total * sizeof(double));
    cudaMalloc((void**)&d_P_m3, N_total * sizeof(double));
    cudaMalloc((void**)&d_P_m4, N_total * sizeof(double));
	cudaMemcpy(d_P_next, h_P_next, N_total * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_P, h_P, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P_m1, h_P_m1, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P_m2, h_P_m2, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P_m3, h_P_m3, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P_m4, h_P_m4, N_total * sizeof(double), cudaMemcpyHostToDevice);
    
    double* d_R1 = new double[N_total];
    double* d_R1m1 = new double[N_total];
    double* d_R1m2 = new double[N_total];
    double* d_R2 = new double[N_total];
    double* d_R2m1 = new double[N_total];
    double* d_R2m2 = new double[N_total];
    cudaMalloc((void**)&d_R1, N_total * sizeof(double));
	cudaMalloc((void**)&d_R1m1, N_total * sizeof(double));
    cudaMalloc((void**)&d_R1m2, N_total * sizeof(double));
    cudaMalloc((void**)&d_R2, N_total * sizeof(double));
    cudaMalloc((void**)&d_R2m1, N_total * sizeof(double));
    cudaMalloc((void**)&d_R2m2, N_total * sizeof(double));
	cudaMemcpy(d_R1, h_R1, N_total * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_R1m1, h_R1m1, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R1m2, h_R1m2, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R2, h_R2, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R2m1, h_R2m1, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R2m2, h_R2m2, N_total * sizeof(double), cudaMemcpyHostToDevice);


	/* for GPU */
	dim3 threads(8, 8, 8); // (x, y, z) (where z = focal axis)
    int block_xy = (Nxy + threads.x - 1) / threads.x;
    int block_z = (Nz + threads.z - 1) / threads.z;
	dim3 blocks(block_xy, block_xy, block_z); // (x, y, z) (where z = focal axis)

    printf("\n N_total = %d; threads = %dx%dx%d; blocks = %dx%dx%d\n", N_total, threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z);
    
    
    //~~~~~~~~~~~~~~~ WESTERVELT EQUATION ~~~~~~~~~~~~~~~~~//
    
    printf("\n WESTERVELT EQUATION ...\n");
    
    double t = 0.0;
    int step = 0;
    while ( t < t_max )
    {        
        Westervelt_space6 <<< blocks, threads >>> (d_P_next, d_P, d_P_m1, d_P_m2, d_P_m3, d_P_m4, d_R1, d_R1m1, d_R1m2, d_R2, d_R2m1, d_R2m2, Nxy, Nz);
        cudaDeviceSynchronize();
        
        step++;
        t = t + h_dt;
        
        boundary_condition_source <<< blocks, threads >>> (d_P_next, t, Nxy, Nz);
        cudaDeviceSynchronize();
        
        boundary_condition_space6 <<< blocks, threads >>> (d_P_next, d_P, t, Nxy, Nz);
        cudaDeviceSynchronize();
        
        printf("\n step %d: t = %f / %f (update...)", step, t, t_max);
        
        // update (destanation, source, ...)
        cudaMemcpy(d_P_m4, d_P_m3, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P_m3, d_P_m2, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P_m2, d_P_m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P_m1, d_P, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P, d_P_next, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        
        cudaMemcpy(d_R1m2, d_R1m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_R1m1, d_R1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_R2m2, d_R2m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_R2m1, d_R2, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    // stop the timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float gputime;
	cudaEventElapsedTime(&gputime, start, stop);
	gputime = gputime / 1000.0;
	printf("\n Processing time for GPU (with data copy): %f (s) \n", gputime);
    
    // start the timer
	cudaEventRecord(start, 0);
    
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SAVING IN CPU MEMORY ~~~~~~~~~~~~~~~~~~~~~~~~~~//
    double time_165p = t;
    cudaMemcpy(h_P_m4, d_P_m4, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P_m3, d_P_m3, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P_m2, d_P_m2, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P_m1, d_P_m1, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P, d_P, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P_next, d_P_next, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(h_R1, d_R1, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R1m1, d_R1m1, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R1m2, d_R1m2, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R2, d_R2, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R2m1, d_R2m1, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_R2m2, d_R2m2, N_total * sizeof(double), cudaMemcpyDeviceToHost);

    
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ACOUSTIC POWER IN WATER ~~~~~~~~~~~~~~~~~~~~~~~~~~//
    printf("\n Acoustic power in water (1 period) ...");
    double* h_Q = new double[N_total];
    for (int idx = 0; idx < N_total; idx++)
        h_Q[idx] = 0.0;
    
    double* d_Q;
    cudaMalloc((void**)&d_Q, N_total * sizeof(double));
    cudaMemcpy(d_Q, h_Q, N_total * sizeof(double), cudaMemcpyHostToDevice);
    
    step = 0;
    while (t < t_max + 1/h_fr) // 1 periods
    {
        Westervelt_space6 <<< blocks, threads >>> (d_P_next, d_P, d_P_m1, d_P_m2, d_P_m3, d_P_m4, d_R1, d_R1m1, d_R1m2, d_R2, d_R2m1, d_R2m2, Nxy, Nz);
        cudaDeviceSynchronize();
        
        step++;
        t = t + h_dt;
        
        boundary_condition_source <<< blocks, threads >>> (d_P_next, t, Nxy, Nz);
        cudaDeviceSynchronize();
        
        boundary_condition_space6 <<< blocks, threads >>> (d_P_next, d_P, t, Nxy, Nz);
        cudaDeviceSynchronize();
        
        acoustic_power_water <<< blocks, threads >>> (d_Q, d_P_next, d_P_m1, Nxy, Nz, false);
        cudaDeviceSynchronize();
        
        printf("\n step %d: t = %f / %f (update...)", step, t, t_max);
        
        // update (destanation, source, ...)
        cudaMemcpy(d_P_m4, d_P_m3, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P_m3, d_P_m2, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P_m2, d_P_m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P_m1, d_P, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_P, d_P_next, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        
        cudaMemcpy(d_R1m2, d_R1m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_R1m1, d_R1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_R2m2, d_R2m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_R2m1, d_R2, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    acoustic_power_water <<< blocks, threads >>> (d_Q, d_P_next, d_P_m1, Nxy, Nz, true);
    cudaDeviceSynchronize();
    
    
    
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FFT and ACOUSTIC POWER IN TISSUE ~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // 500 => for 5 periods; 1000 => for 10 periods; 50000 => for 50 periods
    // since time step = 0.01/fr => 100 steps for 1 period
    int periods = 4;
    printf("\n Westervelt %d periods and FFT for acoustic power in tissue ... \n", periods);
    int N_time = periods * 100; // for # periods
    
    int N_harm = 10; // total number of harmonics;
    
    // fft only for tissue, when z >= 0.12
    int iz_tissue = 1202; // when tissue starts and z >= 0.12
    int Nz_tissue = (Nz - iz_tissue); // 801
    int nx = 6; // 3; // 6; 
    int Nxyz_tissue = nx * Nxy * Nz_tissue; 
    int Nxyz_time = Nxyz_tissue * N_time;
    
    double* d_Pxyz_time;
    cufftComplex* d_Pxyz_c;
    cudaMalloc((void**)&d_Pxyz_time, Nxyz_time * sizeof(double));
    cudaMalloc((void**)&d_Pxyz_c, Nxyz_time * sizeof(cufftComplex));
    
    // Create plan for FFT
    cufftHandle plan_fft;
    cufftPlanMany(&plan_fft, 1, &N_time, NULL, 1, N_time, NULL, 1, N_time, CUFFT_C2C, Nxyz_tissue);
    
    for (int ix = 0; ix < Nxy / nx; ix++) // Nxy / 3 = 142; // Nxy / 6 = 71
    {
        t = time_165p; 
        cudaMemcpy(d_P_next, h_P_next, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_P, h_P, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_P_m1, h_P_m1, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_P_m2, h_P_m2, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_P_m3, h_P_m3, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_P_m4, h_P_m4, N_total * sizeof(double), cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_R1, h_R1, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R1m1, h_R1m1, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R1m2, h_R1m2, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R2, h_R2, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R2m1, h_R2m1, N_total * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R2m2, h_R2m2, N_total * sizeof(double), cudaMemcpyHostToDevice);
        
        printf("\n ix = %d / %d ... ", ix + 1, Nxy / nx);
        for (int idx_time = 0; idx_time < N_time; idx_time++) // Westervelt for # periods 
        {
            if (idx_time == 0 || idx_time % 100 == 99)
                printf("\n step %d / %d (ix = %d)", idx_time+1, N_time, ix);
            
            Westervelt_space6 <<< blocks, threads >>> (d_P_next, d_P, d_P_m1, d_P_m2, d_P_m3, d_P_m4, d_R1, d_R1m1, d_R1m2, d_R2, d_R2m1, d_R2m2, Nxy, Nz);
            cudaDeviceSynchronize();
            
            t = t + h_dt;
            
            boundary_condition_source <<< blocks, threads >>> (d_P_next, t, Nxy, Nz);
            cudaDeviceSynchronize();
            
            boundary_condition_space6 <<< blocks, threads >>> (d_P_next, d_P, t, Nxy, Nz);
            cudaDeviceSynchronize();

            copy_data <<< blocks, threads >>> (d_P_next, d_Pxyz_time, ix, nx, Nxy, Nz_tissue, iz_tissue, idx_time, N_time);
            cudaDeviceSynchronize();

            // update (destanation, source, ...)
            cudaMemcpy(d_P_m4, d_P_m3, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_P_m3, d_P_m2, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_P_m2, d_P_m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_P_m1, d_P, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_P, d_P_next, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            
            cudaMemcpy(d_R1m2, d_R1m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_R1m1, d_R1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_R2m2, d_R2m1, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_R2m1, d_R2, N_total * sizeof(double), cudaMemcpyDeviceToDevice);
        } // time
        printf(" \n Westervelt is done, calculation of FFT and Q ... ");
        
        // FFT
        Double_to_Complex <<< blocks, threads >>> (d_Pxyz_time, d_Pxyz_c, nx, Nxy, Nz_tissue, N_time);
        cudaDeviceSynchronize();
        
        cufftExecC2C(plan_fft, d_Pxyz_c, d_Pxyz_c, CUFFT_FORWARD);
        cudaDeviceSynchronize();
        
        Abs_Complex <<< blocks, threads >>> (d_Pxyz_c, d_Pxyz_time, nx, Nxy, Nz_tissue, N_time);
        cudaDeviceSynchronize();
        
        // Calculate energy Q
        total_energy_Nharmonics_xyz <<< blocks, threads >>> (d_Q, d_Pxyz_time, ix, nx, Nxy, Nz_tissue, iz_tissue, N_time, periods, N_harm);
        cudaDeviceSynchronize();

        //printf(" FFT is done");    

    } // ix
    
    cudaMemcpy(h_Q, d_Q, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    
    // stop the timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gputime, start, stop);
	gputime = gputime / 1000.0;
	printf("\n Processing time for GPU (with data copy and FFT): %f (s) \n", gputime);
    
    std::ofstream file_Q;
    file_Q.open("Q_Z_E100_space6_att9_abs7_2_c1600_relax_fft_v3.dat");
    file_Q << "z(m)" << ";" << "Q(W/m^3)";
    for (int iz = 2; iz < Nz; iz++)
    {
        idx = iz*Nxy*Nxy + 0 + 0 * Nxy; // global index (x=y=0) //
        file_Q << "\n" << h_z_min + iz*h_dz << ";" << h_Q[idx];
    }
    file_Q.close();
    
    
    file_Q.open("Q3D_E100_space6_att9_abs7_2_c1600_relax_fft_v3.bin", std::ios::binary);
    for (int iz = 2; iz < Nz; iz++) // iz == 2  =>  Z = 0
	{
		for (int iy = 0; iy < Nxy; iy++) // rows
		{
			for (int ix = 0; ix < Nxy; ix++) // columns
			{
				idx = iz * (Nxy*Nxy) + ix + iy * Nxy;
				file_Q.write((char*)(&h_Q[idx]), sizeof(double));
			} // ix
		} // iy
	} // iz
    file_Q.close();
    
    
    
 
	// destroy the timer
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    
    
    // free memory
    delete[] h_P_next;
    delete[] h_P;
    delete[] h_P_m1;
    delete[] h_P_m2;
    delete[] h_P_m3;
    delete[] h_P_m4;
    
    delete[] h_R1;
    delete[] h_R1m1;
    delete[] h_R1m2;
    delete[] h_R2;
    delete[] h_R2m1;
    delete[] h_R2m2;
    
    delete[] h_Q;
    
	cudaFree(d_P_next);
    cudaFree(d_P);
    cudaFree(d_P_m1);
    cudaFree(d_P_m2);
    cudaFree(d_P_m3);
    cudaFree(d_P_m4);
    
    cudaFree(d_R1);
    cudaFree(d_R1m1);
    cudaFree(d_R1m2);
    cudaFree(d_R2);
    cudaFree(d_R2m1);
    cudaFree(d_R2m2);
    
    cudaFree(d_Q);
    
    cudaFree(d_Pxyz_time);
    cudaFree(d_Pxyz_c);
    cudaFree(d_Q);

    cufftDestroy(plan_fft);
    
	cudaDeviceReset();
    
	printf("\n Finished \n");
	return 0;
}