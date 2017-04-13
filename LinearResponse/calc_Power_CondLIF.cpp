/*
 * MEX implementation of:
 *	power = calc_Power_cuLIF(w,E0,sigma,tau_ref,v_reset,v_th,tau_m,vlb,dv,rate,Esyn,gE,sigE,Isyn,gI,sigI);
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
 *  Purpose: Calculate the power-spectrum of an LIF neuron driven by 
 *       - current-based white noise
 *       - conductance-based white noise (both E and I)
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
 *  power - Power spectrum (1/ms) evaluated at frequencies w.
 *
 *
 *
 *  Note:
 *
 *  Written following exactly the methods of Richardson "Spike train spectra
 *  and network response..." (2008).
 */

#include "mex.h"
#include "math.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Variable declarations
    int N,i,j,length_w,m;
    
    double E0,sigma,tau_ref,v_reset,v_th,tau_m,vlb,dv;
    double v,pw_r,rate;
    double *pf_r,*pf_i,*jf_r,*jf_i,*p0_r,*p0_i,*j0_r,*j0_i,*w,*power;

    mxArray *pf_mat, *jf_mat, *p0_mat,*j0_mat,*power_mat;

    // parameters for E,I conductances
    double Esyn,gE,sigE,sigE2,Isyn,gI,sigI,sigI2;

    // To save vectors which must be computed for each w
    double sighatsq;
    double *G, *inv_sighatsq;
    mxArray *G_mat, *inv_sighatsq_mat;
    
    // Check # of times injection BC is applied
    int num_apply_inj_bc;
   
    // Retrieve all input parameters.
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
    plhs[0] = mxCreateDoubleMatrix(length_w, 1, mxREAL);
    power = mxGetPr(plhs[0]);
    
    
    N = ((int)((v_th-vlb)/dv)) + 1;
   
    sigE2 = sigE*sigE;
    sigI2 = sigI*sigI;  

    pf_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    pf_r = mxGetPr(pf_mat);
    pf_i = mxGetPi(pf_mat);
    
    jf_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    jf_r = mxGetPr(jf_mat);
    jf_i = mxGetPi(jf_mat);
    
    p0_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    p0_r = mxGetPr(p0_mat);
    p0_i = mxGetPi(p0_mat);
    
    j0_mat = mxCreateDoubleMatrix(N, 1, mxCOMPLEX);
    j0_r = mxGetPr(j0_mat);
    j0_i = mxGetPi(j0_mat);
    
    // V-dep Stuff for integration
    inv_sighatsq_mat = mxCreateDoubleMatrix(N, 1, mxREAL);
    inv_sighatsq     = mxGetPr(inv_sighatsq_mat);

    G_mat = mxCreateDoubleMatrix(N, 1, mxREAL);
    G     = mxGetPr(G_mat);

    // To save computation: set up
    // inv_sighatsq, G beforehand
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
        pf_r[N-1] = 0;
        pf_i[N-1] = 0;
        jf_r[N-1] = 1;
        jf_i[N-1] = 0;
        
        p0_r[N-1] = 0;
        p0_i[N-1] = 0;
        j0_r[N-1] = 0;
        j0_i[N-1] = 0;
        
        num_apply_inj_bc = 0;
        
        v = v_th;
    
        // Solve the IVP.
        for(j = N-1;j >= 1; j--){
	 
            
            jf_r[j-1] = jf_r[j] - dv*w[i]*2*M_PI*pf_i[j];
            jf_i[j-1] = jf_i[j] + dv*w[i]*2*M_PI*pf_r[j];
            
            pf_r[j-1] = pf_r[j]*exp(dv*G[j]) + tau_m*jf_r[j]*inv_sighatsq[j]*(exp(dv*G[j])-1)/G[j];
            pf_i[j-1] = pf_i[j]*exp(dv*G[j]) + tau_m*jf_i[j]*inv_sighatsq[j]*(exp(dv*G[j])-1)/G[j];
            
	    // Check reinjection BC
	    //  This loop must be accessed exactly once!
            if(fabs(v-v_reset)<dv/2){
                j0_r[j-1] = j0_r[j] - dv*w[i]*2*M_PI*p0_i[j] - cos(-w[i]*2*M_PI*tau_ref);
                j0_i[j-1] = j0_i[j] + dv*w[i]*2*M_PI*p0_r[j] - sin(-w[i]*2*M_PI*tau_ref);
                num_apply_inj_bc++; 
            }
            else{
                j0_r[j-1] = j0_r[j] - dv*w[i]*2*M_PI*p0_i[j];
                j0_i[j-1] = j0_i[j] + dv*w[i]*2*M_PI*p0_r[j];
            }
            
            p0_r[j-1] = p0_r[j]*exp(dv*G[j]) + tau_m*j0_r[j]*inv_sighatsq[j]*(exp(dv*G[j])-1)/G[j];
            p0_i[j-1] = p0_i[j]*exp(dv*G[j]) + tau_m*j0_i[j]*inv_sighatsq[j]*(exp(dv*G[j])-1)/G[j];
            
            v = v-dv;
        }
        // The following line implements, in compact form (Eqns. from Richardson 2008):
        //
        // f(w) = -j0(vlb)/jf(Vlb)      (eqn. 21)
        // p(w) = f(w)/(1-f(w))         (eqn. 22)
        // -> and: Re[p(w)]             (needed for eqn. 28)
	
        pw_r = (-(pow(j0_r[0],2) + pow(j0_i[0],2)) - j0_r[0]*jf_r[0] - j0_i[0]*jf_i[0])/(pow(j0_r[0] + jf_r[0],2) + pow(j0_i[0] + jf_i[0],2));
        
        // C(w) = r0*(1+2 Re[p(w)])      (eqn. 28)
        power[i] = rate*(1 + 2*pw_r);
        
        if (num_apply_inj_bc > 1) {
            mexWarnMsgTxt("Injection BC applied more than once!.");
        }
        else if ( num_apply_inj_bc == 0) {
            mexWarnMsgTxt("Injection BC not applied!.");
        }
        
    }
}
        
        
        
        
