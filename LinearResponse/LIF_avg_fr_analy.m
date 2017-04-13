function [ fr ] = LIF_avg_fr_analy( E0, sigma, tau_ref, v_reset, vth, tau_m )
%%
%    LIF_avg_fr_analy: average firing rate of an LIF forced w/ white noise
%                       (approximate analytical expression)
%   
%    tau_m dV = -(V - E0) dt + sigma * sqrt(tau_m) dWt
%     
%    OUTPUTS:   fr:    firing rate (1/ms)
%
%    INPUTS:    E0:         resting potential
%               sigma:      noise std
%               tau_ref:    refractory period (ms)
%               v_reset:    v_reset
%               vth:        threshold
%               tau_m:      membrane time constant (ms)
   
        
        fr= 1/(tau_m*integral(@(x)erfcx(x),(E0-vth)/sigma,(E0-v_reset)/sigma)*sqrt(pi) + tau_ref);

end

