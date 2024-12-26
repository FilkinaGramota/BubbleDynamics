
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>


__device__ double rho; // tissue density, kg/m^3
__device__ double P0; // static background pressure, Pa
__device__ double S2; // surface tension, kg/s^2 (2*S)
__device__ double C_inf; // speed of sound, m/s
__device__ double lambda; // tissue relaxation time, s
__device__ double n; // const in the Tait EoS
__device__ double B; // const in the Tait EoS
__device__ double gamma1; // specific heat ratio for the bubble's interior
__device__ double gamma3; // specific heat ratio for the bubble's interior (3*gamma)
__device__ double mu4; // viscosity, Pa*s (4*mu)
__device__ double G4_3; // tissue elasticity, Pa (4*G/3);


__device__ bool triple_fr; // true => use triple-frequency; false => single (fr[0])


__device__ double minA; // minimal value of Amplitude range
__device__ double stepA; // step for Amplitude

__device__ double minR0; // minimal value of initial radius range
__device__ double stepR0; // step for initial radius





// varying driving sound field (if not triple mode => only fr1 is used)
__device__ double P(const double t, const double Amplitude, const double fr1, const double fr2, const double fr3)
{
	if (!triple_fr)
		return Amplitude * sin(2.0 * M_PI * fr1 * t);
    
    // fr1 == fr2 == fr3 => single mode
    if (fr1 == fr2 && fr1 == fr3)
        return Amplitude * sin(2.0 * M_PI * fr1 * t);
    else
    {
        // fr1 == fr2 or fr2 == fr3 (fr1 != fr3) => dual mode
        if (fr1 == fr2 || fr2 == fr3)
            return (Amplitude / sqrt(2.0)) * (sin(2.0 * M_PI * fr1 * t) + sin(2.0 * M_PI * fr3 * t));
        
        // fr1 != fr2 != fr3 => triple mode
        return (Amplitude / sqrt(3.0)) * (sin(2.0 * M_PI * fr1 * t) + sin(2.0 * M_PI * fr2 * t) + sin(2.0 * M_PI * fr3 * t));
    }
}


// derivative of acoustic signal (sound field) (if not triple mode => only fr1 is used)
__device__ double P_dt(const double t, const double Amplitude, const double fr1, const double fr2, const double fr3)
{
	if (!triple_fr)
		return Amplitude * 2.0 * M_PI * fr1 * cos(2.0 * M_PI * fr1 * t);

    // fr1 == fr2 == fr3 => single mode
    if (fr1 == fr2 && fr1 == fr3)
        return Amplitude * 2.0 * M_PI * fr1 * cos(2.0 * M_PI * fr1 * t);
    else
    {
        // fr1 == fr2 or fr2 == fr3=> dual mode
        if (fr1 == fr2 || fr2 == fr3)
            return Amplitude * sqrt(2.0) * M_PI * (fr1 * cos(2.0 * M_PI * fr1 * t) + fr3 * cos(2.0 * M_PI * fr3 * t));
        
        // fr1 != fr2 != fr3 => triple mode
        return Amplitude * (2.0/sqrt(3.0)) * M_PI * (fr1 * cos(2.0 * M_PI * fr1 * t) + fr2 * cos(2.0 * M_PI * fr2 * t) + fr3 * cos(2.0 * M_PI * fr3 * t));
    }
}


// pressure of gas inside the bubble
__device__ double P_in(const double R, const double R0)
{
	return (P0 + (S2 / R0)) * pow(R0 / R, gamma3);
}


// derivative of the pressure of gas inside the bubble
__device__ double dP_in_dt(const double R0, const double R, const double R1)
{
	return (P0 + (S2 / R0)) * pow(R0, gamma3) * (-gamma3) * pow(R, -gamma3 - 1.0) * R1;
}


// local sound speed at the bubble wall
__device__ double LocalSoundSpeed(const double t, const double R, const double Tau, const double R0, const double Amplitude, const double fr1, const double fr2, const double fr3)
{
	double p = P(t, Amplitude, fr1, fr2, fr3);
	double p_in = P_in(R, R0);
	double C = ((p_in - (S2 / R) + Tau) + B) / (P0 + p + B);
	C = C_inf * pow(C, (n - 1.0) / (2.0 * n));
	return C;
}


__device__ double Stress_dt(const double R, const double R1, const double Tau, const double R0)
{
	double Tau_dt = -G4_3 * (1.0 - pow(R0 / R, 3)) - (mu4 * R1 / R) - Tau;
	return Tau_dt / lambda;
}


__device__ double Q_dt(const double R, const double R1, const double Tau, const double q, const double R0)
{
	double q_dt = (-G4_3 * (1.0 - pow(R0 / R, 3)) - (mu4 * R1 / R)) / (3.0 * lambda);
	return q_dt - (q / lambda) - (R1 * Tau / R);
}


// Enthalpy = H
__device__ double Enthalpy(const double t, const double R, const double Tau, const double R0, const double Amplitude, const double fr1, const double fr2, const double fr3)
{
	double p = P(t, Amplitude, fr1, fr2, fr3);
	double p_in = P_in(R, R0);
	double part1 = (P0 + p + B) * n / (rho * (n - 1.0));
	double part2 = (p_in - (S2 / R) + Tau + B) / (P0 + p + B);
	return part1 * (pow(part2, (n - 1.0) / n) - 1.0);
}


// dH / dt 
__device__ double Enthalpy_dt(const double t, const double R, const double R1, const double Tau, const double Tau1, const double R0, const double Amplitude, const double fr1, const double fr2, const double fr3)
{
	// H = part1 * part2 =>
	// dH/dt = part1' * part2 + part1 * part2'
	double p = P(t, Amplitude, fr1, fr2, fr3);
	double p_in = P_in(R, R0);
	double part1 = (P0 + p + B) * n / (rho * (n - 1.0));
	double p_dt = P_dt(t, Amplitude, fr1, fr2, fr3);
	double part1_dt = p_dt * n / (rho * (n - 1.0));

	// part2 = {I}^(n-1/n) - 1
	// where I is big fraction and I = u / v
	double u = p_in - (S2 / R) + Tau + B;
	double v = P0 + p + B;
	double I = u / v;
	double part2 = pow(I, (n - 1.0) / n) - 1.0;

	double d_pin = dP_in_dt(R0, R, R1);
	double u_dt = d_pin + (S2 * R1 / (R*R)) + Tau1;
	double v_dt = p_dt;
	double I_dt = (u_dt * v - u * v_dt) / pow(v, 2);

	double part2_dt = ((n - 1.0) / n) * pow(I, -1.0 / n) * I_dt;

	return part1_dt * part2 + part1 * part2_dt;
}


// second derivative of radius (d^2R/dt^2)
__device__ double F_R(const double t, const double R, const double R1, const double Tau, const double q, const double R0, const double Amplitude, const double fr1, const double fr2, const double fr3)
{
	double Tau1 = Stress_dt(R, R1, Tau, R0);
	double q1 = Q_dt(R, R1, Tau, q, R0);
	double H = Enthalpy(t, R, Tau, R0, Amplitude, fr1, fr2, fr3);
	double H1 = Enthalpy_dt(t, R, R1, Tau, Tau1, R0, Amplitude, fr1, fr2, fr3);
	double C = LocalSoundSpeed(t, R, Tau, R0, Amplitude, fr1, fr2, fr3);


	double a = 1.0 - (R1 / C);
	double b = 1.0 + (R1 / C);

	double part1 = 1.5 * pow(R1, 2) * (1.0 - (R1 / (3.0 * C)));
	double part2 = b * (H + ((-Tau + 3.0 * q) / rho));
	double part3 = (H1 * a + ((-Tau1 + 3.0 * q1) / rho)) * R / C;
	double f = (-part1 + part2 + part3) / (a * R);

	return f;
}



__device__ double norm_Euclid(const double* error, const double* sc, const int length)
{
	double S = 0.0;
	for (int i = 0; i < length; i++)
		S = S + pow(error[i] / sc[i], 2);
	S = S / length;
	S = sqrt(S);

	return S;
}


__device__ double norm_max(const double* error, const double* sc, const int length)
{
	double max = fabs(error[0] / sc[0]);
    double S = 0;
	for (int i = 1; i < length; i++)
    {
		S = fabs(error[i] / sc[i]);
        if (S > max)
            max = S;
	}

	return max;
}




__device__ void DPRK_step(double &t, double &R, double &R1, double &Tau, double &q, double &h, bool &threshold, const double R0, const double Amplitude, const double fr1, const double fr2, const double fr3, const double endT)
{
	double abs_eps = pow(10.0, -9);
	double rel_eps = pow(10.0, -9);

	// The Butcher tableau (fourth-fifth order Dormand-Prince Runge-Kutta method)
	double a[7][7] = { {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
							{0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
							{3.0 / 40.0, -9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0},
							{44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0},
							{19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0},
							{9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0, 0.0, 0.0},
							{35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0} };
	double b[7] = { 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0 }; // RK 4
	double bb[7] = { 5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0 }; // RK 5
	double c[7] = { 0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0 };

	double Y[4]; // 0=R; 1=dR/dt=R1; 2=Tau; 3=q
	double error[4];
	double sc[4]; 

	double K[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; // for R
	double L[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; // for dR/dt
	double Tau_K[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; // for Tau
	double q_K[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }; // for q

	double k_R, k_R1, k_Tau, k_q, err, max_result;

	if (t + h > endT)
		h = endT - t;

	err = 2.0;
	double fac_max = 5.0;
	threshold = false;
	while (err > 1.0)
	{
		Y[0] = R; Y[1] = R1; Y[2] = Tau; Y[3] = q;
		error[0] = 0.0; error[1] = 0.0; error[2] = 0.0; error[3] = 0.0;
        for (int i = 0; i < 7; i++)
        {
            K[i] = 0.0;
            L[i] = 0.0;
            Tau_K[i] = 0.0;
            q_K[i] = 0.0;
        }
        
		// find dR/dt, R, Tau, q
		for (int i = 0; i < 7; i++)
		{
			k_R = 0.0; k_R1 = 0.0; k_Tau = 0.0; k_q = 0.0;
			for (int j = 0; j < i; j++)
			{
				k_R = k_R + a[i][j] * K[j] * h;
				k_R1 = k_R1 + a[i][j] * L[j] * h;
				k_Tau = k_Tau + a[i][j] * Tau_K[j] * h;
				k_q = k_q + a[i][j] * q_K[j] * h;
			}

			K[i] = (R1 + k_R1);
			L[i] = F_R(t + h * c[i], R + k_R, R1 + k_R1, Tau + k_Tau, q + k_q, R0, Amplitude, fr1, fr2, fr3);
			Tau_K[i] = Stress_dt(R + k_R, R1 + k_R1, Tau + k_Tau, R0);
			q_K[i] = Q_dt(R + k_R, R1 + k_R1, Tau + k_Tau, q + k_q, R0);

			// result
			Y[0] = Y[0] + b[i] * K[i] * h;
			Y[1] = Y[1] + b[i] * L[i] * h;
			Y[2] = Y[2] + b[i] * Tau_K[i] * h;
			Y[3] = Y[3] + b[i] * q_K[i] * h;

			// estimate the error
			error[0] = error[0] + K[i] * h * (b[i] - bb[i]); // R
			error[1] = error[1] + L[i] * h * (b[i] - bb[i]); // dR/dt
			error[2] = error[2] + Tau_K[i] * h * (b[i] - bb[i]); // Tau
			error[3] = error[3] + q_K[i] * h * (b[i] - bb[i]); // q
		}

		// special value (scale)
		sc[0] = abs_eps + rel_eps * fmax(fabs(R), fabs(Y[0]));
		sc[1] = abs_eps + rel_eps * fmax(fabs(R1), fabs(Y[1]));
		sc[2] = abs_eps + rel_eps * fmax(fabs(Tau), fabs(Y[2]));
		sc[3] = abs_eps + rel_eps * fmax(fabs(q), fabs(Y[3]));

		err = norm_Euclid(error, sc, 4);
		//err = norm_max(error, sc, 4);
        max_result = fmax(0.1, 0.9 * pow(err, -0.2));
		if (err > 1.0) // repeat this step
		{
			fac_max = 0.9;
			h = h * fmin(fac_max, max_result);
		}
	}

	t = t + h;
	R = Y[0];
	R1 = Y[1];
	Tau = Y[2];
	q = Y[3];

	h = h * fmin(fac_max, max_result);

	// check threshold
	if (Y[1] <= -C_inf) // Y[1] <= -340.0 // Y[0] >= (2.0 * R0)
	{
		threshold = true;
	}

}



// N_total = N_fr_combo * N_R0
__global__ void DPRK_solver_threshold_fix_tissue(double* results_A, double* d_Fr_combo, const int N_fr_combo, const int N_R0, const int N_A)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    int i_fr_combo = idx / N_R0;
	int i_R0 = idx % N_R0;

	if (i_fr_combo >= 0 && i_fr_combo < N_fr_combo && i_R0 >=0 && i_R0 < N_R0)
	{        
        double R0 = minR0 + i_R0 * stepR0;
		
		int i_fr1 = 0 + i_fr_combo * 3;
		int i_fr2 = 1 + i_fr_combo * 3;
		int i_fr3 = 2 + i_fr_combo * 3;

		double fr1 = d_Fr_combo[i_fr1];
		double fr2 = d_Fr_combo[i_fr2];
		double fr3 = d_Fr_combo[i_fr3];
		
		double endT = 0.0;
		if (triple_fr)
		{
			if (fr1 <= fr2 && fr1 <= fr3)
				endT = 5.0 / fr1;
			if (fr2 <= fr1 && fr2 <= fr3)
				endT = 5.0 / fr2;
			if (fr3 <= fr1 && fr3 <= fr2)
				endT = 5.0 / fr3;
		}
		else
			endT = 5.0 / fr1;

		double h = pow(10.0, -13);
	

		// if for this Amplitude we cannot reach inertial cavitation (threshold value)
		results_A[idx] = -1.0;

		double t = 0.0;
		double R = R0;
		double R1 = 0.0;
		double Tau = 0.0;
		double q = 0.0;

		bool threshold_5 = false, threshold_4 = true, threshold_3 = true;
		
		double Amplitude_5 = 0.0, Amplitude_4 = 0.0, Amplitude_3 = 0.0; 
        
        // (1) step 0.1 MPa
        while (!threshold_5 && Amplitude_5 < pow(10.0, 7))
        {
            t = 0.0; R = R0; R1 = 0.0; Tau = 0.0; q = 0.0;
            Amplitude_5 = Amplitude_5 + pow(10.0, 5);
            threshold_5 = false;
            
            while (t < endT && !threshold_5)
                DPRK_step(t, R, R1, Tau, q, h, threshold_5, R0, Amplitude_5, fr1, fr2, fr3, endT);
        } // ==> exit when threshold_5 = true
        printf("\n R0 = %.3f; A = %.3f(5); fr1 = %.4f; fr2 = %.4f, fr3 = %.4f \n", R0*pow(10.0, 6), Amplitude_5/pow(10.0, 6), fr1/pow(10.0, 6), fr2/pow(10.0, 6), fr3/pow(10.0, 6));
        
        if (!threshold_5 && Amplitude_5 == pow(10.0, 7))
        {
            results_A[idx] = Amplitude_5;
            printf("\n R0 = %.3f; A = %.3f(max); fr1 = %.4f; fr2 = %.4f, fr3 = %.4f \n", R0*pow(10.0, 6), results_A[idx]/pow(10.0, 6), fr1/pow(10.0, 6), fr2/pow(10.0, 6), fr3/pow(10.0, 6));
        }
        
        // (2) step 0.01 MPa
        Amplitude_4 = Amplitude_5;
        while (threshold_5 && threshold_4)
        {
            h = pow(10.0, -13);
            t = 0.0; R = R0; R1 = 0.0; Tau = 0.0; q = 0.0;
            Amplitude_4 = Amplitude_4 - pow(10.0, 4);
            threshold_4 = false;
            
            while (t < endT && !threshold_4)
                DPRK_step(t, R, R1, Tau, q, h, threshold_4, R0, Amplitude_4, fr1, fr2, fr3, endT);
            
            if (!threshold_4)
            {
                // choose the previous one
                Amplitude_4 = Amplitude_4 + pow(10.0, 4);
            }
        } // ==> exit when threshold_4 = false
        printf("\n R0 = %.3f; A = %.3f(4); fr1 = %.4f; fr2 = %.4f, fr3 = %.4f \n", R0*pow(10.0, 6), Amplitude_4/pow(10.0, 6), fr1/pow(10.0, 6), fr2/pow(10.0, 6), fr3/pow(10.0, 6));
        
        // (3) step 0.001 MPa
        Amplitude_3 = Amplitude_4;
        while (threshold_5 && !threshold_4 && threshold_3)
        {
            h = pow(10.0, -13);
            t = 0.0; R = R0; R1 = 0.0; Tau = 0.0; q = 0.0;
            Amplitude_3 = Amplitude_3 - pow(10.0, 3);
            threshold_3 = false;
            
            while (t < endT && !threshold_3)
                DPRK_step(t, R, R1, Tau, q, h, threshold_3, R0, Amplitude_3, fr1, fr2, fr3, endT);
            
            if (!threshold_3)
            {
                results_A[idx] = Amplitude_3 + pow(10.0, 3);
                printf("\n R0 = %.3f; A = %.3f(3); fr1 = %.4f; fr2 = %.4f, fr3 = %.4f \n", R0*pow(10.0, 6), results_A[idx]/pow(10.0, 6), fr1/pow(10.0, 6), fr2/pow(10.0, 6), fr3/pow(10.0, 6));
            }
        } // ==> exit when threshold_3 = false
        
	}
}



int main()
{
	/* SET DEVICE*/
	cudaSetDevice(0);


	// initialize constant variables for global device memory 
	double h_rho = 1100.0; // tissue density, kg/m^3
	double h_P0 = 1.013 * pow(10.0, 5); // static background pressure, Pa
	double h_S2 = 2.0 * 0.056; // surface tension, kg/s^2 (2*S)
	double h_mu4 = 4.0 * 0.009; // 0.009 // viscosity, Pa*s
	double h_G4_3 = 0.04 * pow(10.0, 6) * 4.0 / 3.0; // 0.04 MPa // G*4/3 // elasticity, Pa
	double h_C_inf = 1549.0; // speed of sound, m/s (1549.0; liver)
	double h_lambda = 3.0 * pow(10.0, -9); // tissue relaxation time, s
	double h_n = 7.0; // param from Tait equation of state
    double h_B = (h_C_inf * h_C_inf * h_rho / h_n) - h_P0; // param from Tait equation of state
	double h_gamma = 1.4; // specific heat ratio for the bubble's interior
	double h_gamma3 = 3.0 * h_gamma; // specific heat ratio for the bubble's interior (3*gamma)

	double h_minA = pow(10.0, 4); // minimal value of Amplitude range
	double maxA = pow(10.0, 7); // maximal value of Amplitude range
	double h_stepA = 1000.0; // (maxA - h_minA) / (N_A - 1.0); // step for Amplitude
	int N_A = 1 + (int)((maxA - h_minA) / h_stepA); // 9991 // number of steps for Amplitude 

	double h_stepR0 = pow(10.0, -7); // (maxR0 - h_minR0) / (N_R0 - 1.0); // step for initial radius
    double h_minR0 = pow(10.0, -7); // minimal value of initial radius range
    double maxR0 = pow(10.0, -5); // maximal value of initial radius range
	int N_R0 = 1 + (int)((maxR0 - h_minR0) / h_stepR0); // 100; // number of steps for initial radius

	// triple => fr1 != fr2 ! fr3; single = > use fr1 
	bool h_triple_fr = true;

    double min_fr = 0.02 * pow(10.0, 6); // min value of frequency range
    double max_fr = 5.0 * pow(10.0, 6); // max value of frequency range
    int N_fr = 55; // size of frequency range
	int N_fr_combo = 27775; // 29260 - (repetition of signals) = 27775 // combination with repetition: (N_fr + 3 - 1)! / (3! * (N_fr - 1)!) = 29260
    double step_log = (log10(max_fr) - log10(min_fr)) / (N_fr - 1.0); // step in logarithmic scale   


	// copy from host to global device memory
	cudaMemcpyToSymbol(rho, &h_rho, sizeof(double));
	cudaMemcpyToSymbol(P0, &h_P0, sizeof(double));
	cudaMemcpyToSymbol(S2, &h_S2, sizeof(double));
	cudaMemcpyToSymbol(mu4, &h_mu4, sizeof(double));
	cudaMemcpyToSymbol(G4_3, &h_G4_3, sizeof(double));
	cudaMemcpyToSymbol(C_inf, &h_C_inf, sizeof(double));
	cudaMemcpyToSymbol(lambda, &h_lambda, sizeof(double));
	cudaMemcpyToSymbol(n, &h_n, sizeof(double));
	cudaMemcpyToSymbol(B, &h_B, sizeof(double));
	cudaMemcpyToSymbol(gamma1, &h_gamma, sizeof(double));
	cudaMemcpyToSymbol(gamma3, &h_gamma3, sizeof(double));
	cudaMemcpyToSymbol(triple_fr, &h_triple_fr, sizeof(bool));

	cudaMemcpyToSymbol(minA, &h_minA, sizeof(double));
	cudaMemcpyToSymbol(minR0, &h_minR0, sizeof(double));
	cudaMemcpyToSymbol(stepA, &h_stepA, sizeof(double));
	cudaMemcpyToSymbol(stepR0, &h_stepR0, sizeof(double));


	int N_total = N_fr_combo * N_R0; // total number of simultaniously solving ODE systems 

	// host init memory
	double* h_results = new double[N_total];
	double* h_Fr_combo = new double[N_fr_combo*3];
    
	bool first_cycle = true;
	int i1 = 0, i2 = 0, i3 = 0, i_fr_combo = 0;
	for (int i = 0; i < N_fr; i++)
	{
		first_cycle = true;
        for (int ii = i; ii < N_fr; ii++)
        {
            int iii = ii;
            if (!first_cycle)
                iii = ii + 1;
            for ( ; iii < N_fr; iii++)
            {
                i1 = 0 + i_fr_combo*3;
                i2 = 1 + i_fr_combo*3;
                i3 = 2 + i_fr_combo*3;
                h_Fr_combo[i1] = pow(10.0, log10(min_fr) + i*step_log);
                h_Fr_combo[i2] = pow(10.0, log10(min_fr) + ii*step_log);
                h_Fr_combo[i3] = pow(10.0, log10(min_fr) + iii*step_log);
                i_fr_combo++;
            }
            first_cycle = false;
        }
	}
    
    
    /*
    std::ofstream file_fr;
    file_fr.open("fr3_combo_repeat_N_fr55.dat");
    file_fr << "fr1" << ";" << "fr2" << ";" << "fr3";
    for (i_fr_combo = 0; i_fr_combo < N_fr_combo; i_fr_combo++)
    {
        i1 = 0 + i_fr_combo*3;
        i2 = 1 + i_fr_combo*3;
        i3 = 2 + i_fr_combo*3;
        file_fr << "\n" << h_Fr_combo[i1] << ";" << h_Fr_combo[i2] << ";" << h_Fr_combo[i3];
        if (h_Fr_combo[i1] == h_Fr_combo[i2] && h_Fr_combo[i1] == h_Fr_combo[i3])
            file_fr << ";" << "single";
        else
        {
            if (h_Fr_combo[i1] == h_Fr_combo[i2] || h_Fr_combo[i2] == h_Fr_combo[i3])
                file_fr << ";" << "dual";
        }
    }
    file_fr.close();
    printf("\n Frequency combo is saved ... \n");
    std::cin.get();
    */
    
    
	// device init memory
	double* d_results;
	double* d_Fr_combo;
	cudaMalloc((void**)&d_results, N_total * sizeof(double));
	cudaMalloc((void**)&d_Fr_combo, N_fr_combo*3 * sizeof(double));
	cudaMemcpy(d_Fr_combo, h_Fr_combo, N_fr_combo*3 * sizeof(double), cudaMemcpyHostToDevice);


	/* for GPU */
	int threads = 128; // 64; // pow(2, 3); can vary
	int blocks = (N_total + threads - 1) / threads;
    

	// create the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start the timer
	cudaEventRecord(start, 0);

	printf("\n N_R0 = %d; N_fr_combo = %d; triple_combo = %d \n", N_R0, N_fr_combo, h_triple_fr);
    printf("\n N_total = N_fr_combo * N_R0 = %d; threads = %d; blocks = %d \n", N_total, threads, blocks);


	// ~~~~~~ GPU ~~~~~ //
	DPRK_solver_threshold_fix_tissue <<< blocks, threads >>> (d_results, d_Fr_combo, N_fr_combo, N_R0, N_A);
	cudaDeviceSynchronize();
	cudaMemcpy(h_results, d_results, N_total * sizeof(double), cudaMemcpyDeviceToHost);
	// ~~~~~~ GPU ~~~~~ //



    printf("\n \t save all data ...\n");
    
	std::ofstream fout;
    fout.open("dR_Cinf_liver_fr3_combo_all.dat");

    int idx = 0;
	fout << "fr1" << ";" << "fr2" ";" << "fr3 --> R0";
	for (int i_R0 = 0; i_R0 < N_R0; i_R0++)
		fout << ";" << h_minR0 + i_R0 * h_stepR0;

	for (int i_fr_combo = 0; i_fr_combo < N_fr_combo; i_fr_combo++)
	{
		i1 = 0 + i_fr_combo * 3;
		i2 = 1 + i_fr_combo * 3;
		i3 = 2 + i_fr_combo * 3;
		fout << "\n" << h_Fr_combo[i1] << ";" << h_Fr_combo[i2] << ";" << h_Fr_combo[i3];
		for (int i_R0 = 0; i_R0 < N_R0; i_R0++)
		{
			idx = i_R0 + i_fr_combo * N_R0;
            fout << ";" << h_results[idx];
		} // i_R0
	} // i_fr_combo   


	// stop the timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float gputime;
	cudaEventElapsedTime(&gputime, start, stop);
	gputime = gputime / 1000.0;
    
    fout << "\n" << "time (s)" << ";" << gputime;
    fout.close();

	// destroy the timer
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("\n Processing time for GPU (with data copy): %f (s) \n", gputime);

	// free memory
	delete[] h_results;
	delete[] h_Fr_combo;

	cudaFree(d_results);
	cudaFree(d_Fr_combo);


	cudaDeviceReset();  
    return 0;
}