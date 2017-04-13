/*  
 * 
 * MEX implementation of:
 *	susc = calc_Susc_CondLIF_gE(w,E0,sigma,tau_ref,v_reset,v_th,tau_m,vlb,dv,rate,Esyn,gE,sigE,Isyn,gI,sigI));
 *
 *  Copyright: A.K. Barreiro and C. Ly, 2017
 *  Please cite: Barreiro, A.K. and C. Ly. "When do Correlations Increase with Firing 
 *      Rates in Recurrent Networks?" PLoS Computational Biology, 2017.
 * 
 *  Adapted from code provided by the authors of:
 *     Trousdale J, Hu Y, Shea-Brown E, Josic K. "Impact of network structure and cellular 
 *         response on spike time correlations." PLoS Computational Biology. 2012;8(3):e1002408–e1002408. 
 *
 *
 *  Purpose: Calculate the power-spectra susceptibility of an LIF neuron driven by 
 *       - current-based white noise
 *       - conductance-based white noise (both E and I)
 *     ... w.r.t. changes in the mean EXCITATORY conductance (gE)
 *
 *
 *  Methods:
 *
 *  Follows the threshold integration method described in Richardson "Spike train spectra
 *  and network response functions for nonlinear integrate-and-fire neurons." 
 *  Biological Cybernetics 99: 381-392, 2008.
 *
 *  Neuron dynamics are:
 *     tau_m dV = -(V-E0) dt  - (gE + sigE dW_E)(V-Esyn) -  (gI + sigI dW_I)(V-Isyn) + sqrt(2 tau_m) sigma dW
 *
 *  dW_E, dW_I, and dW are independent white noises.
 *  Notes:
 *    -  the current noise is assumed to be scaled as in Richardson (by sqrt(2 tau)); the conductance noise is not)
 *    -  you do not have to pick v_th-v_reset to be a multiple of dv; however, there will be an 
 *         additional discretization error (O(dv)) if you do not.
 *    - Voltage can be in mV or dimensionless.
 *
 *  Inputs:
 *
 *  w - Array of temporal frequencies (kHz) at which to calculate the spectrum.
 *      Note that the theory diverges at w=0, so zero should be replaced with a
 *      small, but positive value.
*  E0 - Effective rest potential in the absence of noise or any inputs (typically the leak potential + noise mean).
 *  sigma - White noise variance.
 *  tau_ref - Absolute refractory period (ms).
 *  v_reset - Reset potential following the emission of a spike.
 *  v_th - Hard threshold. Upon reaching this threshold, the membrane
 *         potential is reset to v_reset and held there for a fixed amount of time tau_ref.
 *  tau_m - Membrane potential for the neuron (ms).
 *  vlb - Lower bound of the membrane potential. Should be set sufficiently low, 
 *        that it does not impact calculations.
 *  dv - Membrane potential step used in solving the system of BVPs for the desired
 *       statistic.
 *  rate - Stationary firing rate (kHz), calculated from a separate routine.
 *  Esyn   - Excitatory reversal potential
 *  gE     - Mean E conductance
 *  sigE   - sigma for E conductance
 *  Isyn   - Inhibitory reversal potential
 *  gI     - Mean I conductance
 *  sigI   - sigma for I condunctance
 *
 *  Output:
 *
 *  susc - Susceptibility (i.e. the Fourier transform of the linear response function) 
 *       evaluated at frequencies w. 
 *
 */

#include "mex.h"
#include "math.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Variable declarations
    int N,i,j,length_w,m;
    
    double E0,sigma,tau_ref,v_reset,v_th,tau_m,vlb,dv,v,r0,rate;
    double *p0_r,*p0_i,*j0_r,*j0_i,*pr_r,*pr_i,*jr_r,*jr_i,*pe_r,*pe_i,*je_r,*je_i;
    double *w,*susc_r,*susc_i;
    
    mxArray *p0_mat,*j0_mat,*pr_mat,*jr_mat,*pe_mat,*je_mat,*transfer_mat;
    
    // parameters for E,I conductances
    double Esyn,gE,sigE,sigE2,Isyn,gI,sigI,sigI2;

    // To save vectors which must be recomputed for every w
    double sighatsq;
    double *G, *inv_sighatsq;
    mxArray *G_mat, *inv_sighatsq_mat;

    // Check # of times injection BC is applied
    int num_apply_inj_bc;
    
    // Input parameters
    w=mxGetPr(prhs[0]);
    E0=mxGetScalar(prhs[1]);
    sigma=mxGetScalar(prhs[2]);
    tau_ref=mxGetScalar(prhs[3]);
    v_reset=mxGetScalar(prhs[4]);
    v_th=mxGetScalar(prhs[5]);
    tau_m=mxGetScalar(prhs[6]);
    vlb=mxGetScalar(prhs[7]);
    dv=mxGetScalar(prhs[8]);
    rate=mxGetScalar(prhs[9]);

     //New input parameters
    Esyn=mxGetScalar(prhs[10]);
    gE=mxGetScalar(prhs[11]);
    sigE=mxGetScalar(prhs[12]);
    Isyn=mxGetScalar(prhs[13]);
    gI=mxGetScalar(prhs[14]);
    sigI=mxGetScalar(prhs[15]);

    
    length_w = mxGetN(prhs[0]);
    m = mxGetM(prhs[0]);
    
    if(length_w != 1 && m != 1)
    	mexErrMsgTxt("Frequency vector should be a Nx1 vector.");
    if(length_w < m)
        length_w = m;

    
    // Output parameters
    plhs[0] = mxCreateDoubleMatrix(length_w, 1, mxCOMPLEX);
    susc_r = mxGetPr(plhs[0]);
    susc_i = mxGetPi(plhs[0]);
    
    
    N = ((int)((v_th-vlb)/dv)) + 1;
    
    sigE2 = sigE*sigE;
    sigI2 = sigI*sigI;

    pr_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    pr_r = mxGetPr(pr_mat);
    pr_i = mxGetPi(pr_mat);
    
    jr_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    jr_r = mxGetPr(jr_mat);
    jr_i = mxGetPi(jr_mat);
    
    p0_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    p0_r = mxGetPr(p0_mat);
    p0_i = mxGetPi(p0_mat);
    
    j0_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    j0_r = mxGetPr(j0_mat);
    j0_i = mxGetPi(j0_mat);
    
    pe_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    pe_r = mxGetPr(pe_mat);
    pe_i = mxGetPi(pe_mat);
    
    je_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    je_r = mxGetPr(je_mat);
    je_i = mxGetPi(je_mat);
    
    inv_sighatsq_mat = mxCreateDoubleMatrix(N, 1, mxREAL);
    inv_sighatsq     = mxGetPr(inv_sighatsq_mat);
    G_mat = mxCreateDoubleMatrix(N, 1, mxREAL);
    G     = mxGetPr(G_mat);

    // To save computation: set up
    // inv_sighatsq, G beforehand
    // (Both are v-dependent)
    v     = v_th;
    for(j = N-1;j >= 1; j--){
        
        sighatsq = sigma*sigma + (sigE2*(v-Esyn)*(v-Esyn) + sigI2*(v-Isyn)*(v-Isyn))/(2*tau_m);
	inv_sighatsq[j] = 1/sighatsq;

	G[j] = (v-E0 + (v-Esyn)*(gE+sigE2/tau_m) + (v-Isyn)*(gI+sigI2/tau_m))*inv_sighatsq[j];
	//Make sure G ~= 0)
	if (fabs(G[j])<1e-10)
	   G[j] = 1e-10;
	v = v-dv;
    }

    // For each frequency value, solve an IVP for the spectrum at that
    // frequency.
    for(i = 0; i < length_w; i++){
        
        // Set the final conditions for the IVP.
        pr_r[N-1] = 0;
        pr_i[N-1] = 0;
        jr_r[N-1] = 1;
        jr_i[N-1] = 0;
        
        p0_r[N-1] = 0;
        p0_i[N-1] = 0;
        j0_r[N-1] = 1;
        j0_i[N-1] = 0;
        
        pe_r[N-1] = 0;
        pe_i[N-1] = 0;
        je_r[N-1] = 0;
        je_i[N-1] = 0;
        
        v = v_th;
        num_apply_inj_bc = 0;
    
        // Solve the IVP.
        for(j = N-1;j >= 1; j--){
	 
            // Iterate j_r/p_r
            // Iterate j_0/p_0
            
	  // Check the reinjection BC
            if(fabs(v-v_reset)<dv/2) {
                jr_r[j-1] = jr_r[j] - dv*w[i]*2*M_PI*pr_i[j] - cos(-w[i]*2*M_PI*tau_ref);
                jr_i[j-1] = jr_i[j] + dv*w[i]*2*M_PI*pr_r[j] - sin(-w[i]*2*M_PI*tau_ref);
                j0_r[j-1] = j0_r[j] - 1;
                num_apply_inj_bc++;
            }
            else {
                jr_r[j-1] = jr_r[j] - dv*w[i]*2*M_PI*pr_i[j];
                jr_i[j-1] = jr_i[j] + dv*w[i]*2*M_PI*pr_r[j];
                j0_r[j-1] = j0_r[j];
            }
            pr_r[j-1] = pr_r[j]*exp(dv*G[j]) + tau_m*inv_sighatsq[j]*jr_r[j]*(exp(dv*G[j])-1)/G[j];
            pr_i[j-1] = pr_i[j]*exp(dv*G[j]) + tau_m*inv_sighatsq[j]*jr_i[j]*(exp(dv*G[j])-1)/G[j];
            
            p0_r[j-1] = p0_r[j]*exp(dv*G[j]) + tau_m*inv_sighatsq[j]*j0_r[j]*(exp(dv*G[j])-1)/G[j];
            p0_i[j-1] = p0_i[j]*exp(dv*G[j]) + tau_m*inv_sighatsq[j]*j0_i[j]*(exp(dv*G[j])-1)/G[j];
            
       
            
            // Iterate j_E/p_E
            
            je_r[j-1] = je_r[j] - dv*w[i]*2*M_PI*pe_i[j];
            je_i[j-1] = je_i[j] + dv*w[i]*2*M_PI*pe_r[j];
            
            //use "rate*p0_r" because p0_r has not yet been scaled (i.e. P0 = rate*p0)
	    // FROM cuLIF...   H =tau_m*J - P0
            //pe_r[j-1] = pe_r[j]*exp(dv*G[j]) + (tau_m*je_r[j]-rate*p0_r[j])*inv_sigmasq*(exp(dv*G)-1)/G;
            //pe_i[j-1] = pe_i[j]*exp(dv*G[j]) + (tau_m*je_i[j]-rate*p0_i[j])*inv_sigmasq*(exp(dv*G)-1)/G;

	    // For CondLIF: H = tau_m*J + (v-Esyn)*P0
	    pe_r[j-1] = pe_r[j]*exp(dv*G[j]) + (tau_m*je_r[j]-(Esyn-v)*rate*p0_r[j])*inv_sighatsq[j]*(exp(dv*G[j])-1)/G[j];
            pe_i[j-1] = pe_i[j]*exp(dv*G[j]) + (tau_m*je_i[j]-(Esyn-v)*rate*p0_i[j])*inv_sighatsq[j]*(exp(dv*G[j])-1)/G[j];
            
            v = v-dv;
        }
        
        if (num_apply_inj_bc > 1) {
            mexWarnMsgTxt("Injection BC applied more than once!.");
        }
        else if ( num_apply_inj_bc == 0) {
            mexWarnMsgTxt("Injection BC not applied!.");
        }
        
        susc_r[i] = -(je_r[0]*jr_r[0] + je_i[0]*jr_i[0])/(pow(jr_r[0],2) + pow(jr_i[0],2));
        susc_i[i] = (je_r[0]*jr_i[0] - je_i[0]*jr_r[0])/(pow(jr_r[0],2) + pow(jr_i[0],2));   
    }
}
