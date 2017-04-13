function [ rates_C,cov_sc_Cvar,spk_corr_Cvar] = LR_CondBased_fn( paramfile, thresh_h, sigmas, runpm, neurpm )
% LR_CondBased_Fn: Apply network linear response to a coupled 
%           E/I network of conductance-based LIF neurons
%
% Inputs:   paramfile: 
%                Contains:   W_ee, W_ie, etc..:   connectivity lists
%                            g_vec            :   connectivity strengths
%           thresh_h:  Neural thresholds (unscaled)
%           sigmas: noise terms
%           runpm [OPTIONAL]: parameters for running the algorithm: dw, T, etc..
%           neurpm [OPTIONAL]: parameters for network and cells: t_re, t_de, etc..
%     
% Outputs:  rates_C:     vector of steady-state firing rates
%           cov_sc_Cvar: spike count covariance matrix (for specified Twin)
%           spk_corr_Cvar: spike count correlation matrix (for specified
%                               Twin)
%
%           size(cov_sc_Cvar) = size(spk_corr_Cvar) = [N N length(Twin)]
%
%
% Copyright: A.K. Barreiro and C. Ly, 2017
%  Please cite: Barreiro, A.K. and C. Ly. "When do Correlations Increase with Firing 
%      Rates in Recurrent Networks?" PLoS Computational Biology, 2017.
% 
%  Adapted from code provided by the authors of:
%     Trousdale J, Hu Y, Shea-Brown E, Josic K. "Impact of network structure and cellular 
%          response on spike time correlations." PLoS Computational Biology. 2012;8(3):e1002408?e1002408. 
%
%

%% Default run parameters
Tmax = 200;         % Maximum time lag over which to calculate cross-correlations (ms)
dt = 0.5;           % Bin size for which to calculate cross-correlations (ms)
vlb = -10;         % Lower bound on membrane potential to use in threshold integration
dv = 10E-4;         % Membrane potential step to use in threshold integration 

Twin = [5;50;100];  % desired time windows for covariances

%% Process run parameters
if isfield(runpm,'Tmax'); Tmax = runpm.Tmax; end
if isfield(runpm,'dt'); dt = runpm.dt; end
if isfield(runpm,'vlb'); vlb = runpm.vlb; end
if isfield(runpm,'dv'); dv = runpm.dv; end
if isfield(runpm,'Twin'); Twin = runpm.Twin; end    

% Generate a vector of frequencies at which to solve for the spectral
% statistics in order to generate cross-correlations with maximum lag Tmax
% and bin size dt.
dw = 1/2/Tmax;
wmax = 1/2/dt;
w = -wmax:dw:(wmax-dw);
ind0 = find(abs(w) < 1e-8);
w(ind0) = 1e-8;
bins = length(w);



%% Default neural/network parameters
Ne=80; %number of E-cells
Ni=20; %number of I-cells
N = Ne+Ni;

E0 = 0;             % Effective rest potential (mV)
tau_ref = 2;        % Absolute refractory period (ms)
v_reset = 0;        % Post-spike reset potential (mV)
v_th = 1;           % Spiking threshold (mV)
tau_m = 20;         % Membrane time constant (ms)
Esyn = 6.5;           % Excit reversal potential (mV)
Isyn = -.5;           % Inhib reversal potential (mV)

ampE=1;             % Amplitude of conductance jumps from E connections (dimensionless); 
t_re=1;             % Rise time of excitatory conductance (ms)
t_de=5;             % Decay time of excitatory conductance (ms)

ampI=2;             % Amplitude of conductance jumps from I connections (dimensionless);        
t_ri=2;             % Rise time of inhibitory conductance (ms)
t_di=10;            % Decay time of inhibitory conductance (ms)

%% Process network/neural parameters
if isfield(neurpm,'Ne'); Ne = neurpm.Ne; end
if isfield(neurpm,'Ni'); Ni = neurpm.Ni; end
if isfield(neurpm,'E0'); E0 = neurpm.E0; end
if isfield(neurpm,'tau_ref'); tau_ref = neurpm.tau_ref; end
if isfield(neurpm,'v_reset'); v_reset = neurpm.v_reset; end
if isfield(neurpm,'v_th'); v_th = neurpm.v_th; end
if isfield(neurpm,'tau_m'); tau_m = neurpm.tau_m; end
if isfield(neurpm,'Esyn'); Esyn = neurpm.Esyn; end
if isfield(neurpm,'Isyn'); Isyn = neurpm.Isyn; end
if isfield(neurpm,'t_re'); t_re = neurpm.t_re; end
if isfield(neurpm,'t_de'); t_de = neurpm.t_de; end
if isfield(neurpm,'ampE'); ampE = neurpm.ampE; end
if isfield(neurpm,'t_ri'); t_ri = neurpm.t_ri; end
if isfield(neurpm,'t_di'); t_di = neurpm.t_di; end
if isfield(neurpm,'ampI'); ampI = neurpm.ampI; end

N=Ne+Ni; %total number of cells
taud = zeros(N,1);       % Synaptic delays (ms)

%% Load connectivity info from parameter file
% 
load(paramfile,'W_*');
load(paramfile,'g_vec');

% Was colored or white noise used?
colored_noise_flag = 0;
% Sigma must be:
% - Multiplied by (v_th - v_reset), for voltage scaling
% 
% IF colored noise is used, also:
% - Divided by sqrt(2) because variance of colored noise is actually 1/2
%
%   (Note: division by tau_m will be done in loop; since usually teff will be
%   used instead)
if (colored_noise_flag)
    sigmas = sigmas/sqrt(2);
end

% Scale  thresholds ampE/ampI, sigmas by voltage
vth_h   = thresh_h*(v_th - v_reset) + v_reset;
ampE    = ampE*(v_th-v_reset);
ampI    = ampI*(v_th-v_reset);
sigmas = sigmas*(v_th-v_reset);


%% Create connectivity matrices
C=zeros(N,N); 

% Cvar: Connection matrix for computing effective noise variance in
% fixed-point iteration (for rates) and for susceptibility wrt conducntance
% noise variances
Cvar    = C;

for j=1:Ne
    cl_indx=W_ee(j,:)+1;
    cl_indx=cl_indx(cl_indx>0);
    
    C(j,cl_indx)=g_vec(3)*(ampE*t_re);
    Cvar(j,cl_indx) = (g_vec(3)*ampE)^2*t_re/2.;
    
    cl_indx=W_ei(j,:)+1;
    cl_indx=cl_indx(cl_indx>0);
    
    C(j,Ne+cl_indx)=g_vec(1)*(ampI*t_ri);
    Cvar(j,Ne+cl_indx)=(g_vec(1)*ampI)^2*t_ri/2.;
end

% For second order alpha fns, multiply by factor tr/(tr+td)
Cvar(1:Ne,1:Ne) = Cvar(1:Ne,1:Ne)*(t_re/(t_re+t_de));
Cvar(1:Ne,Ne+1:N) = Cvar(1:Ne,Ne+1:N)*(t_ri/(t_ri+t_di));

for j=Ne+1:N
    cl_indx=W_ie(j-Ne,:)+1;
    cl_indx=cl_indx(cl_indx>0);
    
    C(j,cl_indx)=g_vec(2)*(ampE*t_re);
    Cvar(j,cl_indx)=(g_vec(2)*ampE)^2*t_re/2.;
    
    cl_indx=W_ii(j-Ne,:)+1;
    cl_indx=cl_indx(cl_indx>0);
    
    C(j,Ne+cl_indx)=g_vec(4)*(ampI*t_ri);
    Cvar(j,Ne+cl_indx)=(g_vec(4)*ampI)^2*t_ri/2.;
end

% For second order alpha fns, multiply by factor tr/(tr+td)
Cvar(Ne+1:N,1:Ne) = Cvar(Ne+1:N,1:Ne)*(t_re/(t_re+t_de));
Cvar(Ne+1:N,Ne+1:N) = Cvar(Ne+1:N,Ne+1:N)*(t_ri/(t_ri+t_di));


%% Solve for firing rates by fixed point iteration
%% Generate an initial estimate for the fixed-point iteration
r0=zeros(N,1);
for j=1:N
    %   Call LIF_avg_fr_analy w/
    %       time const = tau_m
    %       potential  = E0
    %       noise std  = sigma/sqrt(tau_m)
    r0(j,1)  = LIF_avg_fr_analy( E0, sigmas(j)/sqrt(tau_m), tau_ref, v_reset, vth_h(j), tau_m );
end

rates_C = r0;
rates_temp_C = zeros(N,1);

max_num_rate_fp_its = 200;
%% Iterate until convergence
for i = 1:max_num_rate_fp_its
    for j = 1:N
        
        % Mean conductances
        gEmean = C(j,1:Ne)*rates_C(1:Ne);
        gImean = C(j,Ne+1:N)*rates_C(Ne+1:end);
        
        % Stdev. of each conductance 
        sigE   = sqrt(Cvar(j,1:Ne)*rates_C(1:Ne));
        sigI   = sqrt(Cvar(j,Ne+1:N)*rates_C(Ne+1:N));
        
        rates_temp_C(j) = calc_Rate_CondLIF(E0,sigmas(j)/sqrt(2*tau_m),tau_ref,...
            v_reset,vth_h(j),tau_m,vlb,dv,...
            Esyn,gEmean,sigE,Isyn,gImean,sigI);
           
    end
    
    % Compare rates w/ rates_temp: are we sufficiently converged?
    if ( norm(rates_C-rates_temp_C)/(max(norm(rates_C),norm(rates_temp_C))) < 0.001)
        break;
    end
        
    rates_C = rates_temp_C;
end


% Calculate the uncoupled power spectrum, susceptibility, and Fourier
% transformed synaptic kernel for every cell in the network at the
% frequency values w.
Ft = zeros(N,bins);

% Power spectrum
Ct0_C = zeros(N,bins);

% Four total susceptibility fns must be computed
% Susceptibility w.r.t. gE and gI
AtE_C  = zeros(N,bins);
AtI_C  = AtE_C;

% Susceptibility w.r.t. sigE and sigI 
AtE_Cvar  = AtE_C;
AtI_Cvar  = AtE_C;

for i = 1:N
    
    %%%% Conductance-based
    gEmean = C(i,1:Ne)*rates_C(1:Ne);
    gImean = C(i,Ne+1:N)*rates_C(Ne+1:end);

    sigE   = sqrt(Cvar(i,1:Ne)*rates_C(1:Ne));
    sigI   = sqrt(Cvar(i,Ne+1:N)*rates_C(Ne+1:N));
   
    
    Ct0_C(i,:)  = calc_Power_CondLIF(w,E0,sigmas(i)/sqrt(2*tau_m),...
        tau_ref,v_reset,vth_h(i),tau_m,vlb,dv,rates_C(i),...
        Esyn,gEmean,sigE,Isyn,gImean,sigI);
    
    AtE_C(i,:) = calc_Susc_CondLIF_gE(w,E0,sigmas(i)/sqrt(2*tau_m),...
        tau_ref,v_reset,vth_h(i),tau_m,vlb,dv,rates_C(i),...
        Esyn,gEmean,sigE,Isyn,gImean,sigI);
    
    AtI_C(i,:) = calc_Susc_CondLIF_gI(w,E0,sigmas(i)/sqrt(2*tau_m),...
        tau_ref,v_reset,vth_h(i),tau_m,vlb,dv,rates_C(i),...
        Esyn,gEmean,sigE,Isyn,gImean,sigI);
    
    
    AtE_Cvar(i,:) = calc_Susc_CondLIF_sigE2(w,E0,sigmas(i)/sqrt(2*tau_m),...
        tau_ref,v_reset,vth_h(i),tau_m,vlb,dv,rates_C(i),...
        Esyn,gEmean,sigE,Isyn,gImean,sigI);
    
    AtI_Cvar(i,:) = calc_Susc_CondLIF_sigI2(w,E0,sigmas(i)/sqrt(2*tau_m),...
        tau_ref,v_reset,vth_h(i),tau_m,vlb,dv,rates_C(i),...
        Esyn,gEmean,sigE,Isyn,gImean,sigI);
    
    % Calculate the values of Fourier transform of the synaptic kernel for cell i.
    % Note: Each kernel must be normalized to area 1.
    %       ("C" already contains raw area under synaptic kernel)
    
    for j = 1:bins
        % single time constant alpha function (from Trousdale, et al. 2012)
        % Ft(i,j)=exp(1i*-2*pi*w(j)*taud(i))/((1-1i*-2*pi*w(j)*taus(i))^2);
       
        if(i<=Ne) %first Ne are E-cells
            Ft(i,j) = (1/ampE/t_re)*...
                exp(1i*-2*pi*w(j)*taud(i))*ampE/(t_de/t_re-1)*(1/(1/t_de-1i*-2*pi*w(j)) - 1/(1/t_re-1i*-2*pi*w(j)));
            
            % UNNORMALIZED synaptic kernel?
            %Ft(i,j) = exp(1i*-2*pi*w(j)*taud(i))/(t_de-t_re)*(1/(1/t_de-1i*-2*pi*w(j)) - 1/(1/t_re-1i*-2*pi*w(j)));
     
        else
            Ft(i,j) = (1/ampI/t_ri)*...
                exp(1i*-2*pi*w(j)*taud(i))*ampI/(t_di/t_ri-1)*(1/(1/t_di-1i*-2*pi*w(j)) - 1/(1/t_ri-1i*-2*pi*w(j)));
            
            % UNNORMALIZED synaptic kernel?
            %Ft(i,j) = exp(1i*-2*pi*w(j)*taud(i))/(t_di-t_ri)*(1/(1/t_di-1i*-2*pi*w(j)) - 1/(1/t_ri-1i*-2*pi*w(j)));
        end
    end
end 

% Solve the linear response equations for the auto- and cross-spectra of
% every pair of cells in the network and store in yy.
I = eye(N);

yy_C = zeros(N,N,bins);
K_C  = zeros(N,N);
yy0_C = zeros(N,N);

for j = 1:bins
    for k = 1:N
        for l = 1:N            
            if (l <= Ne)
                % Susceptibility to E inputs (mean)
                K_C(k,l) = C(k,l)*AtE_C(k,j)*Ft(l,j);
            else
                % Susceptibility to I inputs (mean)
                K_C(k,l) = C(k,l)*AtI_C(k,j)*Ft(l,j);
            end
            
            % Include contribution from change to variance.
            if (l <= Ne)
                K_C(k,l) = K_C(k,l) + Cvar(k,l)*AtE_Cvar(k,j)*Ft(l,j);
            else
                K_C(k,l) = K_C(k,l) + Cvar(k,l)*AtI_Cvar(k,j)*Ft(l,j);
            end
        end
        yy0_C(k,k) = Ct0_C(k,j);
    end
    
    % NOTE: No need to save K. K can be easily reconstructed later at desired frequency:
    %    K = diag(At(:,f))*C*diag(Ft(:,f))
    
    yy_C(:,:,j) = (I-K_C)\yy0_C/(I-K_C');

end

%% Keep power of unperturbed cells
yy0keep_C = zeros(size(Ct0_C));

for i=1:N    
    [temp1,temp2]=inv_f_trans_on_vector(w,squeeze(Ct0_C(i,:)));
    yy0keep_C(i,:) = real(temp2);    
end

% Calculate the inverse Fourier transform of the auto-/cross-spectra for
% every pair and store the results back in yy.
for i = 1:N
    for j = 1:N
        [t_th,temp_ccg] = inv_f_trans_on_vector(w,squeeze(yy_C(i,j,:)));
        yy_C(i,j,:) = real(temp_ccg);
    end
end

%% Calculate Cov/Var/Corr for selected time windows 

% First, calculate kernels

% t_th returned by inv_f_trans_on_vector
d_th=t_th(2)-t_th(1);
kern_corr=zeros(length(t_th),length(Twin));
for k=1:length(Twin)
    kern_temp=t_th'; %assuming t_th same (w is same)
 
    kern_temp(t_th<0)=Twin(k)+t_th(t_th<0);
    kern_temp(t_th>=0)=Twin(k)-t_th(t_th>=0);
    kern_temp(abs(t_th)>Twin(k))=0; %outside of triang=0
    kern_corr(:,k)=kern_temp;
end

% Integrate by kernels: do all kernels at once
cov_sc_Cvar=ones(N,N,length(Twin));     % Covariance/variance
spk_corr_Cvar=ones(N,N,length(Twin));   % Pearson's Correlation 

for i = 1:N
    cov_sc_Cvar(i,i,:)=sum(kern_corr.*repmat(squeeze(yy_C(i,i,:)),1,length(Twin)))*d_th;
    for j = (i+1):N    
        cov_sc_Cvar(i,j,:) = sum(kern_corr.*repmat(squeeze(yy_C(i,j,:)),1,length(Twin)))*d_th; 
                %don't need to make length(t_th)x1, it's 1 x length(t_th)
        spk_corr_Cvar(i,j,:) = squeeze(cov_sc_Cvar(i,j,:))./sqrt(squeeze(cov_sc_Cvar(i,i,:)) ...
            .*( sum(kern_corr.*repmat(squeeze(yy_C(j,j,:)),1,length(Twin))) )'*d_th); 
        cov_sc_Cvar(j,i,:) = cov_sc_Cvar(i,j,:);     %symmetric
        spk_corr_Cvar(j,i,:) = spk_corr_Cvar(i,j,:); %symmetric
    end
end



%% Default save parameter
save_LR_output_flag = 0;

%% Process save parameters
if (isfield(runpm,'save_LR_output_flag')) 
    save_LR_output_flag = runpm.save_LR_output_flag;
    if (save_LR_output_flag)
        try
            LR_outfile =runpm.LR_outfile;
        catch
            warning('runpm: LR_outfile not found. Cannot save results');
            save_LR_output_flag = 0;
        end
    end
end

% Save output
if (save_LR_output_flag)

    % File names and parameters
    save(LR_outfile,'colored_noise_flag','paramfile','sigmas','vth_h');
    save(LR_outfile,'Tmax','dt','Ne','Ni','E0','tau_ref','v_reset','v_th','tau_m','Esyn','Isyn','-append');
    save(LR_outfile,'ampE','ampI','t_re','t_de','t_ri','t_di','-append');
    
    % THIS WOULD RESULT IN A LARGE FILE
    % LEAVE OUT FOR NOW
    % Results: rates, unperturbed autocorrs, auto/xcorr fns.
    %save(LR_outfile,'yy0keep_C','yy_C','-append');
      
    % Make K0: Use this to analyze contributions from graph motifs
    CenterBin = ceil(Tmax/dt)+1;
    j=CenterBin;
    for k=1:N
        for l=1:N
            if (l <= Ne)
                % Susceptibility to E inputs
                K_C(k,l) = C(k,l)*AtE_C(k,j)*Ft(l,j);
            else
                K_C(k,l) = C(k,l)*AtI_C(k,j)*Ft(l,j);
            end
            % Include contribution from change to variance as well!
            if (l <= Ne)
                K_C(k,l) = K_C(k,l) + Cvar(k,l)*AtE_Cvar(k,j)*Ft(l,j);
            else
                K_C(k,l) = K_C(k,l) + Cvar(k,l)*AtI_Cvar(k,j)*Ft(l,j);
            end
        end
    end
    K0 = K_C;
      
    save(LR_outfile,'Ct0_C','K0','-append');
    % Results: Cov and Rho specific T
    save(LR_outfile,'rates_C','Twin','cov_sc_Cvar','spk_corr_Cvar','-append');
end

end

