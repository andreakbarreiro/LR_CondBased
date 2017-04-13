/*  
 * MEX implementation of:
 *	rate = calc_Rate_CondLIF(E0,sigma,tau_ref,v_reset,v_th,tau_m,vlb,dv,Esyn,gE,sigE,Isyn,gI,sigI);
 *
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
 *  Purpose: Calculate the stationary firing rate of an LIF neuron driven by 
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
 *    -  you do not have to pick v_th-vlb to be a multiple of dv; however, there will be an 
 *         additional discretization error (O(dv)) if you do not.
 *    - Voltage can be in mV or dimensionless.
 *
 *
 *  Parameters:
 *
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
 *  Esyn   - Excitatory reversal potential
 *  gE     - Mean E conductance
 *  sigE   - sigma for E conductance  
 *  Isyn   - Inhibitory reversal potential
 *  gI     - Mean I conductance
 *  sigI   - sigma for I condunctance
 *
 *
 *
 *  Output:
 *
 *  rate - Stationary firing rate (kHz).
 *
 *
 */

#include "mex.h"
#include "math.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Variable declarations
    int N,i;
    
    double E0,sigma,tau_ref,v_reset,v_th,tau_m,vlb,dv;
    double sighatsq, inv_sighatsq;
    double sum_p0,v,G;
    double *p0,*j0,*rate;
    
    double Esyn,gE,sigE,sigE2,Isyn,gI,sigI,sigI2;

    // Check # of times injection BC is applied
    int num_apply_inj_bc;
    
    mxArray *p0_mat,*j0_mat;
    
    // Input parameters
    E0=mxGetScalar(prhs[0]);
    sigma=mxGetScalar(prhs[1]);
    tau_ref=mxGetScalar(prhs[2]);
    v_reset=mxGetScalar(prhs[3]);
    v_th=mxGetScalar(prhs[4]);
    tau_m=mxGetScalar(prhs[5]);
    vlb=mxGetScalar(prhs[6]);
    dv=mxGetScalar(prhs[7]);
    
    //New input parameters
    Esyn=mxGetScalar(prhs[8]);
    gE=mxGetScalar(prhs[9]);
    sigE=mxGetScalar(prhs[10]);
    Isyn=mxGetScalar(prhs[11]);
    gI=mxGetScalar(prhs[12]);
    sigI=mxGetScalar(prhs[13]);

    // Output parameters
    plhs[0]=mxCreateDoubleMatrix(1, 1, mxREAL);
    rate=mxGetPr(plhs[0]);
    
    N = ((int)((v_th-vlb)/dv)) + 1;
   
    sigE2 = sigE*sigE;
    sigI2 = sigI*sigI;

    p0_mat = mxCreateDoubleMatrix(N, 1, mxREAL);
    p0 = mxGetPr(p0_mat);
    
    j0_mat = mxCreateDoubleMatrix(N, 1, mxREAL);
    j0 = mxGetPr(j0_mat);
    
    // Set the final conditions for the IVP. 
    p0[N-1] = 0;
    j0[N-1] = 1;
    
    v = v_th;
    
    num_apply_inj_bc = 0;
    
    // Solve the IVP.
    for(i = N-1; i >= 1; --i){
      
      //sig2 needs to be computed for each v!
      sighatsq = sigma*sigma + (sigE2*(v-Esyn)*(v-Esyn) + sigI2*(v-Isyn)*(v-Isyn))/(2*tau_m);

      inv_sighatsq = 1/sighatsq;

      G = (v-E0 + (v-Esyn)*(gE+sigE2/tau_m) + (v-Isyn)*(gI+sigI2/tau_m))*inv_sighatsq;
      //Make sure G ~= 0)
      if (fabs(G)<1e-10)
	 G = 1e-10;  

        //Check for re-injection BC
        if(fabs(v-v_reset) < (dv/2)) {
            j0[i-1] = j0[i] - 1;
            num_apply_inj_bc++;
        }
        else
            j0[i-1] = j0[i];
        
        p0[i-1] = p0[i]*exp(dv*G) + tau_m*j0[i]*inv_sighatsq*(exp(dv*G)-1)/G;
        
        v = v-dv;
    }
    
    if (num_apply_inj_bc > 1) {
        mexWarnMsgTxt("calc_Rate_CondLIF: Injection BC applied more than once!.");
    }
    else if ( num_apply_inj_bc == 0) {
        mexWarnMsgTxt("calc_Rate_CondLIF: Injection BC not applied!.");
    }
    
    sum_p0 = p0[0];
    for(i = 1; i < N; i++)
        sum_p0 += p0[i];
        
    *rate = 1/(dv*sum_p0 + tau_ref);
}
        
        
        
        
