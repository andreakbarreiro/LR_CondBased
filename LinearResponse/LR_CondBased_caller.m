
%
% LR_CondBased_caller.m: Driver script to compute network linear response of a coupled 
%           E/I network of conductance-based LIF neurons
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

% As needed...
mex calc_Rate_CondLIF.cpp
mex calc_Power_CondLIF.cpp
mex calc_Susc_CondLIF_gE.cpp
mex calc_Susc_CondLIF_gI.cpp

mex calc_Susc_CondLIF_sigE2.cpp
mex calc_Susc_CondLIF_sigI2.cpp


%% Parameters: which network to look at?
% het_vs_hom:
%       = 1:        heterogeneous
%       = 0:        homogeneous
het_vs_hom = 1;

% network_state:
%       = 1:        asynchronous
%       = 2:        strong asynchronous
network_state = 2;

% Set # of cells here
Ne = 80; Ni = 20;
N = Ne+Ni;

% Pick parameter file: will be read by routine LR_CondBased_fn
%
% paramfile must contain:
%   W_ee, W_ie, etc..:     connectivity lists  (processed by _fn)
%   g_vec            :     connectivity strengths, in order: [g_ei; g_ie;
%                               g_ee; g_ii]  (processed by _fn)
%   Thres_new        :     vector of thresholds, ordered I cells first
%                           (processed here)
%
% Formatting notes:
%   W_jk, g_jk refers to a k->j connection
%   Each W_jk is given as a nt x k matrix, where: nt=size of target
%                       population, k=# of connections from source
%                       population into EACH target pop. cell (or maximal
%                       #, when this varies from cell-to-cell)
%   Each entry is a C-style ([0, ns-1]) index over the source population
%       (ns is the size of the source population)
%   For example: suppose there are 80 E cells and 20 I cells, and each E
%       cell receives exactly 6 connections. Then W_ei is a 80 x 6 matrix;
%       each row contains an ordered list of integers between 0 and 19.
%   
% Other variables in paramfile are needed for associated Monte Carlo
% simulations
% Also see: create_ic.m        
%

if (network_state == 1)
    regime_str='Asyn'; 
    if (het_vs_hom)
        paramfile='../Examples/asyNcPs_het.mat';
    else
        paramfile='../Examples/asyNcPs_hom.mat';
    end
    sigmas = [.2*ones(Ne,1); .3*ones(Ni,1)];   % Noise std (mV)
elseif (network_state == 2)
    regime_str='StrAs';
    if (het_vs_hom)
        paramfile='../Examples/strAsyPs_het.mat';
    else
        paramfile='../Examples/strAsyPs_hom.mat';
    end
    sigmas = [.15*ones(Ne,1); .25*ones(Ni,1)];   % Noise std (mV)
else
    error('Network state is not recognized...quitting');
end

% In LIF code, time is measured in seconds; here in ms.
% Therefore, sigma scales by sqrt{1 s/1 ms} = sqrt(1000)
sigmas = sigmas*sqrt(1000);

% Adjust thresholds, which are accessed differently in Monte Carlo routine
% (In paramfile, stored I cell thresholds first)
load(paramfile,'Thres_new');

vth_h=[Thres_new(Ni+1:end); Thres_new(1:Ni)]; %switch so E-cells 1st; 


% runpm, neurpm can be empty: only need to include what you
%  want to change
runpm = [];
neurpm = [];

Twin = [5;50;100];
runpm.Twin=Twin;

[ rates_C,cov_sc_Cvar,spk_corr_Cvar] = LR_CondBased_fn( paramfile, vth_h, sigmas, runpm, neurpm );


% Plot all EE correlations vs. firing rates

% Ne*(Ne-1)/2; total number of distinct rho_EE pairs
% mt_ee = [I,J] pairs of C-style subscript indices 
%
mt_ee=[];
for j=1:(Ne-1)
    mt_ee=[mt_ee; (j-1)*ones(Ne-j,1) (j:Ne-1)'];
end
% Transform them to Matlab-style, linear indices
ree_ind=sub2ind([N N],mt_ee(:,1)+1,mt_ee(:,2)+1); %single index for rho_EE pairs in NxN matrix

% Ne x Ni rho_IE pairs
rie_ind=sub2ind([N N], repmat((1:Ne)',Ni,1) , reshape(repmat((Ne+1:N),Ne,1),Ne*Ni,1) );


scrsz = get(groot,'ScreenSize');
figure('Position',[1000 scrsz(4)*2/3 scrsz(3)/2 scrsz(4)/2])

% Geometric mean firing rates
nuE_geom_C=sqrt(rates_C(mt_ee(:,1)+1).*rates_C(mt_ee(:,2)+1))*1000; %so in Hz

% For each time window
for k=1:length(Twin)
    subplot(1,length(Twin),k);
    
    % Pick out which to plot, remove duplicates, and reorder
    r_temp = spk_corr_Cvar(:,:,k);
    r_temp = r_temp(ree_ind);
  
    plot(nuE_geom_C,r_temp,'.','MarkerSize',18);
    set(gca,'FontSize',18)
    title(sprintf('%s, T = %d ms',regime_str,Twin(k)));
    xlabel('Mean Rates (Geom., Hz)')
    ylabel('LR approx. to \rho_{EE}')
end
