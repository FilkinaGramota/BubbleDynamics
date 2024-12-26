#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>


// medium properties
// water
__constant__ double rho; // tissue density, [kg/m^3]
__constant__ double c_inf; // speed of sound

// tissue (muscle, pork)
__constant__ double rho_tissue; // tissue density, [kg/m^3]
__constant__ double c_inf_tissue; // speed of sound


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
__constant__ double dz;
__constant__ double dt_T; // time step for heat equation


__constant__ double kdt_dxy; // dt*k/(dxy*dxy) // for heat equation (in XY direction)
__constant__ double kdt_dz; // dt*k/(dz*dz) // for heat equation (in Z direction)
__constant__ double kdt_dxy_water; // dt*k/(dxy*dxy) // for heat equation in water (in XY direction)
__constant__ double kdt_dz_water; // dt*k/(dz*dz) // for heat equation in water (in Z direction)
__constant__ double dt_rho_ct; // dt/(rho*c_t) // for q_ac
__constant__ double dt_rho_ct_water; // dt/(rho*c_t) // for q_ac in water






__global__ void Temperature_square(double* T_n, double* T_p, double* Q, double* TD, const int Nxy, const int Nz, const bool use_p, const int shift_x, const int shift_y)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // columns
    int iy = threadIdx.y + blockDim.y * blockIdx.y; // rows
	int iz = threadIdx.z + blockDim.z * blockIdx.z; // number of matricies Nx*Ny

	int Nxy_2 = Nxy * Nxy;
	int idx = iz * Nxy_2 + ix + iy * Nxy; // global index
    
    int ix_ac = ix - shift_x;
    int iy_ac = iy - shift_y;
    int idx_ac = iz * Nxy_2 + ix_ac + iy_ac * Nxy;

	if (ix > 0 && ix < Nxy-1 && iy > 0 && iy < Nxy-1 && iz > 0 && iz < Nz-1) // update only interior points
	{
        //~~~~~~~~~~~~~~~~~~ HEAT EQUATION ~~~~~~~~~~~~~~~~~~~//
		int left = iz * Nxy_2 + (ix - 1) + iy * Nxy;
		int right = iz * Nxy_2 + (ix + 1) + iy * Nxy;
		int up = iz * Nxy_2 + ix + (iy + 1) * Nxy;
		int down = iz * Nxy_2 + ix + (iy - 1) * Nxy;
		int zp = (iz + 1) * Nxy_2 + ix + iy * Nxy;
		int zm = (iz - 1) * Nxy_2 + ix + iy * Nxy;
        
        double q_ac = 0.0;
        if (ix_ac >= 0 && ix_ac < Nxy && iy_ac >= 0 && iy_ac < Nxy)
            q_ac = Q[idx_ac];
        
        double Z = iz*dz;
		double T_c = 0.0; // current temperature in celcius        
        
        if (use_p)
        {
            if (Z < water_tissue)
                T_n[idx] = T_p[idx] * (1.0 - 4.0 * kdt_dxy_water - 2.0 * kdt_dz_water) + kdt_dxy_water * (T_p[left] + T_p[right] + T_p[up] + T_p[down]) + kdt_dz_water * (T_p[zp] + T_p[zm]) + dt_rho_ct_water * q_ac;
            else
                T_n[idx] = T_p[idx] * (1.0 - 4.0 * kdt_dxy - 2.0 * kdt_dz) + kdt_dxy * (T_p[left] + T_p[right] + T_p[up] + T_p[down]) + kdt_dz * (T_p[zp] + T_p[zm]) + dt_rho_ct * q_ac;
            T_c = T_n[idx] - 273.15;
        }
		else
        {
            if (Z < water_tissue)
                T_p[idx] = T_n[idx] * (1.0 - 4.0 * kdt_dxy_water - 2.0 * kdt_dz_water) + kdt_dxy_water * (T_n[left] + T_n[right] + T_n[up] + T_n[down]) + kdt_dz_water * (T_n[zp] + T_n[zm]) + dt_rho_ct_water * q_ac;
            else
                T_p[idx] = T_n[idx] * (1.0 - 4.0 * kdt_dxy - 2.0 * kdt_dz) + kdt_dxy * (T_n[left] + T_n[right] + T_n[up] + T_n[down]) + kdt_dz * (T_n[zp] + T_n[zm]) + dt_rho_ct * q_ac;
            T_c = T_p[idx] - 273.15;
        }
        
        if (T_c < 43.0)
            TD[idx] = TD[idx] + pow(4.0, T_c - 43.0) * dt_T;
        else
            TD[idx] = TD[idx] + pow(2.0, T_c - 43.0) * dt_T;
            
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
    
    // tissue (muscle) parameters [Maxim_2014]
    double h_rho_tissue = 1055.0; // density
    double h_c_inf_tissue = 1600.0; // 1600.0; // 1550.0; // speed of sound
    double c_t_tissue = 3700.0; // 3200.0; // tissue specific heat [J/(kg*K)]
	double K_t_tissue = 0.51; // 0.49; // tissue thermal conductivity [W/(m*K)]
	double k_tissue = K_t_tissue / (h_rho_tissue * c_t_tissue);
    
    
    double h_water_tissue = 0.12; // 0.12; 0.115 // where water ends and tissue starts 
    
    double speed = 2.0 * pow(10.0, -3); // speed of transducer movement
    double scale = 3.0 * pow(10.0, -3); // spacing between spiral lines
    
    double speed_init = 0.05 * pow(10.0, -3); // initial speed = 0.05mm/s
    double t_acc = 0.15; // acceleration time (s)
    double acceleration = (speed - speed_init) / t_acc;
    
    // boundary and initial conditions
	double T0 = 23.6 + 273.15; // in Kelvin
    double T0_water = 22.7 + 273.15; // in Kelvin
    
    
    // mesh parameters
    double lambda = c_inf_water / h_fr; // wavelength [m]
    
    double h_xy_min = -0.085, xy_max = 0.085; // [m]
    double h_dxy = 0.0002;
    int Nxy = 1 + round((xy_max - h_xy_min) / h_dxy);
    int Nxy_half = Nxy / 2;
    
    double z_max = 0.2; // focal axis // z_min = 0.0
    double h_dz = 0.0001;
    int Nz = 1 + round(z_max / h_dz);

    
    double h_dt_T = 0.001;
    int Nt_scale = 1 + (int)(scale / (speed * h_dt_T));
    double t_scale = Nt_scale * h_dt_T;
    
    int Nt_acc = 1 + (int)(t_acc / h_dt_T);
    double* speed_acc = new double[Nt_acc];
    for (int i = 0; i < Nt_acc; i++)
        speed_acc[i] = speed_init + i*h_dt_T*acceleration;
    
    printf("\n xy = [%.3f; %.3f]m; Nxy = %d \n", h_xy_min, xy_max, Nxy);
	printf("\n z = [0.0; %.3f]m; Nz = %d \n", z_max, Nz);
    printf("\n dxy = %.4f m; dz = %.4f m \n", h_dxy, h_dz);
	printf("\n dt = %.10f; Nt_scale = %d \n", h_dt_T, Nt_scale);
    printf("\n speed = %.4f [m/s]; scale = %.3f [m] \n", speed, scale);
    printf("\n Water ends at Z = %.3f [m] \n", h_water_tissue);
    
        

	// create the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start the timer
	cudaEventRecord(start, 0);
    

	// copy from host to global device memory    
    cudaMemcpyToSymbol(rho, &rho_water, sizeof(double));
    cudaMemcpyToSymbol(c_inf, &c_inf_water, sizeof(double));
    
    cudaMemcpyToSymbol(rho_tissue, &h_rho_tissue, sizeof(double));
    cudaMemcpyToSymbol(c_inf_tissue, &h_c_inf_tissue, sizeof(double));
    
    cudaMemcpyToSymbol(fr, &h_fr, sizeof(double));
    cudaMemcpyToSymbol(d, &h_d, sizeof(double));
    cudaMemcpyToSymbol(alpha, &h_alpha, sizeof(double));
    cudaMemcpyToSymbol(alpha_hole, &h_alpha_hole, sizeof(double));
    
    cudaMemcpyToSymbol(water_tissue, &h_water_tissue, sizeof(double)); 

    cudaMemcpyToSymbol(xy_min, &h_xy_min, sizeof(double));
    cudaMemcpyToSymbol(dxy, &h_dxy, sizeof(double));
    cudaMemcpyToSymbol(dz, &h_dz, sizeof(double));
    cudaMemcpyToSymbol(dt_T, &h_dt_T, sizeof(double));
    
    double h_kdt_dxy = h_dt_T * k_tissue / (h_dxy * h_dxy);
    double h_kdt_dz = h_dt_T * k_tissue / (h_dz * h_dz);
    double h_kdt_dxy_water = h_dt_T * k_water / (h_dxy * h_dxy);
    double h_kdt_dz_water = h_dt_T * k_water / (h_dz * h_dz);
    double h_dt_rho_ct = h_dt_T / (h_rho_tissue * c_t_tissue);
    double h_dt_rho_ct_water = h_dt_T / (rho_water * c_t_water);
    
	cudaMemcpyToSymbol(kdt_dxy, &h_kdt_dxy, sizeof(double));
    cudaMemcpyToSymbol(kdt_dz, &h_kdt_dz, sizeof(double));
    cudaMemcpyToSymbol(kdt_dxy_water, &h_kdt_dxy_water, sizeof(double));
    cudaMemcpyToSymbol(kdt_dz_water, &h_kdt_dz_water, sizeof(double));
    cudaMemcpyToSymbol(dt_rho_ct, &h_dt_rho_ct, sizeof(double));
    cudaMemcpyToSymbol(dt_rho_ct_water, &h_dt_rho_ct_water, sizeof(double));


	int N_total = Nxy*Nxy * Nz;
    double* h_T_p = new double[N_total];
	double* h_T_n = new double[N_total];
    double* h_TD = new double[N_total];

	// initial and boundary conditions
    int idx = 0;
	for (int iz = 0; iz < Nz; iz++)
	{
		double Z = iz * h_dz;
        for (int iy = 0; iy < Nxy; iy++) // rows
		{
			for (int ix = 0; ix < Nxy; ix++) // columns
			{
				idx = iz * (Nxy*Nxy) + ix + iy * Nxy;
                if (Z < h_water_tissue)
                {
                    h_T_p[idx] = T0_water;
                    h_T_n[idx] = T0_water;
                    h_TD[idx] = pow(4.0, T0_water-273.15 - 43.0) * h_dt_T;
                }
				else
                {
                    h_T_p[idx] = T0;
                    h_T_n[idx] = T0;
                    h_TD[idx] = pow(4.0, T0-273.15 - 43.0) * h_dt_T;
                }
                
			} // ix
		} // iy
	} // iz
    
	double* d_T_p;
	double* d_T_n;
    double* d_TD;
	cudaMalloc((void**)&d_T_p, N_total * sizeof(double));
	cudaMalloc((void**)&d_T_n, N_total * sizeof(double));
    cudaMalloc((void**)&d_TD, N_total * sizeof(double));
	cudaMemcpy(d_T_p, h_T_p, N_total * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_T_n, h_T_n, N_total * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_TD, h_TD, N_total * sizeof(double), cudaMemcpyHostToDevice);
    
	double* h_Q = new double[N_total];
    
    int N_total_small = pow(Nxy_half+1, 2) * Nz;
    double* Q_small = new double[N_total_small];
    
    std::ifstream file_Q;
    printf("\n Reading Q file ... \n");
    file_Q.open("Q3D_E100_space6_att9_abs7_2_c1600_relax_fft_v3.bin", std::ios::binary);
    file_Q.read((char*)Q_small, sizeof(double)*N_total_small);
    file_Q.close();
    
    int ix_small = 0, iy_small = 0, idx_small = 0; 
    for (int iz = 0; iz < Nz; iz++)
    {
        for (int iy = 0; iy < Nxy; iy++)
        {
            if (iy < Nxy_half)
                iy_small = Nxy_half - iy;
            else
                iy_small = iy - Nxy_half;
            for (int ix = 0; ix < Nxy; ix++)
            {
                if (ix < Nxy_half)
                    ix_small = Nxy_half - ix;
                else
                    ix_small = ix - Nxy_half;
                
                idx = (iz * Nxy*Nxy) + ix + iy * Nxy; // global index for Q
                idx_small = (iz * pow(Nxy_half+1, 2)) + ix_small + iy_small * (Nxy_half+1); // global index for Q_small
                
                h_Q[idx] = Q_small[idx_small];
            }
        }
    }
    
    double* d_Q;
    cudaMalloc((void**)&d_Q, N_total * sizeof(double));
    cudaMemcpy(d_Q, h_Q, N_total * sizeof(double), cudaMemcpyHostToDevice);

	

	/* for GPU */
	dim3 threads(8, 8, 8); // (x, y, z) (where z = focal axis)
    int block_xy = (Nxy + threads.x - 1) / threads.x;
    int block_z = (Nz + threads.z - 1) / threads.z;
	dim3 blocks(block_xy, block_xy, block_z); // (x, y, z) (where z = focal axis)

    printf("\n N_total = %d; threads = %dx%dx%d; blocks = %dx%dx%d\n", N_total, threads.x, threads.y, threads.z, blocks.x, blocks.y, blocks.z);
    
    
    // ~~~~~~~~~~~~~~~~~~~ TEMPERATURE ~~~~~~~~~~~~~~~~~~//
    printf("\n BIOHEAT EQUATION ...\n");
    printf("\n speed = %.4f [m/s]; scale = %.3f [m] \n", speed, scale);
    
    bool use_p = true;
    double t = 0.0;
    
    int lines = 0;
    // 5 mm // 3mm
    int spiral_lines = 13; // 13; // 9;
    int increase = 1; //6-7;
    
    double x_s = h_xy_min + 425*h_dxy; // x = 0.0;
    double y_s = h_xy_min + 425*h_dxy; // y = 0.0;
    
    int sign_x = 1, sign_y = -1; // up = -1, down = +1
    int shift_x = 0; // as index
    int shift_y = 0; // as index
    
    bool right_left = true; // start move to right or left
    
    double x_next = x_s, x_prev = x_s;
    double y_next = y_s, y_prev = y_s;
    double diff_next = 0.0, diff_prev = 0.0;
    
    double speed_current = speed_init;
    while (lines < spiral_lines)
    {
        int n = 0;
        if (lines == 0)
            n = 1;

        x_next = x_prev + sign_x*h_dxy;
        y_next = y_prev + sign_y*h_dxy;
        
        for ( ; n < increase*Nt_scale + 2*Nt_acc; n++)
        {
            t = t + h_dt_T;
            if (n < Nt_acc) // increase speed
                speed_current = speed_acc[n];
            if (n >= Nt_acc && n < increase*Nt_scale + Nt_acc) // constant
                speed_current = speed;
            if (n >= increase*Nt_scale + Nt_acc) // decrease
                speed_current = speed_acc[increase*Nt_scale + 2*Nt_acc - n - 1];
                
            if (right_left)
            {
                x_s = x_s + sign_x*speed_current*h_dt_T;
                diff_prev = abs(x_s - x_prev) / h_dxy;
                diff_next = abs(x_next - x_s) / h_dxy;
                if (diff_next < diff_prev)
                {
                    shift_x = shift_x + sign_x;
                    x_prev = x_next;
                    x_next = x_next + sign_x*h_dxy;
                }
            }
            else
            {
                y_s = y_s + sign_y*speed_current*h_dt_T;
                diff_prev = abs(y_s - y_prev) / h_dxy;
                diff_next = abs(y_next - y_s) / h_dxy;
                if (diff_next < diff_prev)
                {
                    shift_y = shift_y + sign_y;
                    y_prev = y_next;
                    y_next = y_next + sign_y*h_dxy;
                }
            }
            
            Temperature_square <<< blocks, threads >>> (d_T_n, d_T_p, d_Q, d_TD, Nxy, Nz, use_p, shift_x, shift_y);
            cudaDeviceSynchronize();
            use_p = !use_p;
        } // for loop

        right_left = !right_left;
        if (right_left)
        {
            increase++;
            sign_x = -1*sign_x;
            x_next = x_prev;
        }
        else
        {
            //increase--;
            sign_y = -1*sign_y;
            y_next = y_prev;
        }
            
        lines++;
        printf("\n line = %d / %d (t = %f)", lines, spiral_lines, t);
        
        /**/
        if (lines == 1)
        {
            if (!use_p)
                cudaMemcpy(h_T_n, d_T_n, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(h_T_n, d_T_p, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            
            std::ofstream file_xz, file_xy;
            file_xz.open("T_XZ_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop0.dat");
            file_xy.open("T_XY_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop0_z15.dat");
            
            // XZ
            file_xz << "Z=rows" << ";" << "X=columns";
            for (int iz = 1200; iz < 1701; iz++) // 12-17 cm
            {
                file_xz << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0-2cm
                {
                    idx = iz * (Nxy*Nxy) + ix + 425 * Nxy; // 1) 355; 2) 425; 3) 425; 4) 567
                    file_xz << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xz.close();
            
            // XY
            file_xy << "Y=rows" << ";" << "X=columns";
            for (int iy = 325; iy < 526; iy++) // -2.0 - 2cm
            {
                file_xy << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0 - 2cm
                {
                    idx = 1500 * (Nxy*Nxy) + ix + iy * Nxy; // 3) 1500 = 15 cm
                    file_xy << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xy.close();
            
        } // lines == 1
        
        if (lines == 5)
        {
            if (!use_p)
                cudaMemcpy(h_T_n, d_T_n, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(h_T_n, d_T_p, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            
            std::ofstream file_xz, file_xy;
            file_xz.open("T_XZ_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop1.dat");
            file_xy.open("T_XY_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop1_z15.dat");
            
            // XZ
            file_xz << "Z=rows" << ";" << "X=columns";
            for (int iz = 1200; iz < 1701; iz++) // 12-17 cm
            {
                file_xz << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0 - 2.0cm
                {
                    idx = iz * (Nxy*Nxy) + ix + 425 * Nxy; // 1) 355; 2) 425; 3) 425; 4) 567
                    file_xz << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xz.close();
            
            // XY
            file_xy << "Y=rows" << ";" << "X=columns";
            for (int iy = 325; iy < 526; iy++) // -2.0 - 2cm
            {
                file_xy << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0 - 2cm
                {
                    idx = 1500 * (Nxy*Nxy) + ix + iy * Nxy; // 3) 1500 = 15 cm
                    file_xy << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xy.close();
        } // lines == 5
        
        if (lines == 9)
        {
            if (!use_p)
                cudaMemcpy(h_T_n, d_T_n, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(h_T_n, d_T_p, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            
            std::ofstream file_xz, file_xy;
            file_xz.open("T_XZ_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop2.dat");
            file_xy.open("T_XY_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop2_z15.dat");
            
            // XZ
            file_xz << "Z=rows" << ";" << "X=columns";
            for (int iz = 1200; iz < 1701; iz++) // 12-17 cm
            {
                file_xz << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0 - 2cm
                {
                    idx = iz * (Nxy*Nxy) + ix + 425 * Nxy; // 1) 355; 2) 425; 3) 425; 4) 567
                    file_xz << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xz.close();
            
            // XY
            file_xy << "Y=rows" << ";" << "X=columns";
            for (int iy = 325; iy < 526; iy++) // -2.0 - 2cm
            {
                file_xy << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0 - 2cm
                {
                    idx = 1500 * (Nxy*Nxy) + ix + iy * Nxy; // 3) 1500 = 15 cm
                    file_xy << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xy.close();
        } // lines == 9
        
        if (lines == 13)
        {
            if (!use_p)
                cudaMemcpy(h_T_n, d_T_n, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            else
                cudaMemcpy(h_T_n, d_T_p, N_total * sizeof(double), cudaMemcpyDeviceToHost);
            
            std::ofstream file_xz, file_xy;
            file_xz.open("T_XZ_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop3.dat");
            file_xy.open("T_XY_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_loop3_z15.dat");
            
            // XZ
            file_xz << "Z=rows" << ";" << "X=columns";
            for (int iz = 1200; iz < 1701; iz++) // 12-17 cm
            {
                file_xz << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0 - 2cm
                {
                    idx = iz * (Nxy*Nxy) + ix + 425 * Nxy; // 1) 355; 2) 425; 3) 425; 4) 567
                    file_xz << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xz.close();
            
            // XY
            file_xy << "Y=rows" << ";" << "X=columns";
            for (int iy = 325; iy < 526; iy++) // -2.0 - 2cm
            {
                file_xy << "\n";
                for (int ix = 325; ix < 526; ix++) // -2.0 - 2cm
                {
                    idx = 1500 * (Nxy*Nxy) + ix + iy * Nxy; // 3) 1500 = 15 cm
                    file_xy << h_T_n[idx] << ";";
                } // iy
            } // iz
            file_xy.close();
        } // lines == 13
        /**/
        
    } // while (lines)
    
    if (!use_p)
        cudaMemcpy(h_T_n, d_T_n, N_total * sizeof(double), cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(h_T_n, d_T_p, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(h_TD, d_TD, N_total * sizeof(double), cudaMemcpyDeviceToHost);
    
    /**/
    std::ofstream file, file2;
    file.open("TD_XZ_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2.dat");
    file << "Z=rows" << ";" << "X=columns"; 
    file << ";" << "speed=" << ";" << h_c_inf_tissue << ";" << "c_t=" << ";" << c_t_tissue << ";" << "K_t=" << ";" << K_t_tissue;
    for (int iz = 1200; iz < 1601; iz++) // [12;16] cm
	{
        file << "\n";
		for (int ix = 325; ix < 527; ix++) // [-2;2.02] cm
		{
            idx = iz * (Nxy*Nxy) + ix + 425 * Nxy; // global index (y = 0)
            file << h_TD[idx] << ";";
		} // ix
	} // iz
    file.close();
    /**/
    
    /**/
    file.open("TD_XY_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_z15.dat");
    file << "Y=rows" << ";" << "X=columns" << ";" << "z=15cm"; 
    file << ";" << "speed=" << ";" << h_c_inf_tissue << ";" << "c_t=" << ";" << c_t_tissue << ";" << "K_t=" << ";" << K_t_tissue;
    for (int iy = 325; iy < 526; iy++) // [-2;2] cm
	{
        file << "\n";
		for (int ix = 325; ix < 527; ix++) // [-2;2.02] cm
		{
            idx = 1500 * (Nxy*Nxy) + ix + iy * Nxy; // global index (y = 0)
            file << h_TD[idx] << ";";
		} // ix
	} // iz
    file.close();
    /**/
    
    /**/
    file.open("TD_XY_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_z13_5.dat");
    file << "Y=rows" << ";" << "X=columns" << ";" << "z=13.5cm"; 
    file << ";" << "speed=" << ";" << h_c_inf_tissue << ";" << "c_t=" << ";" << c_t_tissue << ";" << "K_t=" << ";" << K_t_tissue;
    for (int iy = 325; iy < 526; iy++) // [-2;2] cm
	{
        file << "\n";
		for (int ix = 325; ix < 527; ix++) // [-2;2.02] cm
		{
            idx = 1350 * (Nxy*Nxy) + ix + iy * Nxy; // global index (y = 0)
            file << h_TD[idx] << ";";
		} // ix
	} // iz
    file.close();
    /**/
    
    /**/
    file.open("TD_XY_E100_2mms_3mm_att9_abs7_2_c1600_relax_v2_z12_5.dat");
    file << "Y=rows" << ";" << "X=columns" << ";" << "z=12.5cm"; 
    file << ";" << "speed=" << ";" << h_c_inf_tissue << ";" << "c_t=" << ";" << c_t_tissue << ";" << "K_t=" << ";" << K_t_tissue;
    for (int iy = 325; iy < 526; iy++) // [-2;2] cm
	{
        file << "\n";
		for (int ix = 325; ix < 527; ix++) // [-2;2.02] cm
		{
            idx = 1250 * (Nxy*Nxy) + ix + iy * Nxy; // global index (y = 0)
            file << h_TD[idx] << ";";
		} // ix
	} // iz
    file.close();
    /**/
    
    /*
    //file2.open("T_3D_E100_2mms_3mm_v2_3_abs6_3.bin", std::ios::binary);
    //file.open("TD_3D_E100_2mms_3mm_att9_abs6_3_c1600_relax_v2.bin", std::ios::binary);
    idx = iz * (Nxy*Nxy) + ix + iy * Nxy;
	file.write((char*)(&h_TD[idx]), sizeof(double));
    //file2.write((char*)(&h_T_n[idx]), sizeof(double));
    */
   
   
    // stop the timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float gputime;
	cudaEventElapsedTime(&gputime, start, stop);
	gputime = gputime / 1000.0;
	printf("\n Processing time for GPU (with data copy): %f (s) \n", gputime);
    
 
	// destroy the timer
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    
    // free memory
    delete[] speed_acc;
    
    delete[] h_Q;
    delete[] Q_small;
    
    delete[] h_T_p;
	delete[] h_T_n;
    delete[] h_TD;
    
    cudaFree(d_Q);
    
    cudaFree(d_T_p);
    cudaFree(d_T_n);
    cudaFree(d_TD);

	cudaDeviceReset();
    
	printf("\n Finished \n");
	return 0;
}