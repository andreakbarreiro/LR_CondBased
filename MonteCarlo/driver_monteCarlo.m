%m-file to run the mex function mx_wnNoReflBndry.c

%load inputs, connect, Parameters, etc
load asyNcPs %!! must run creat_ic.m first !!!

a_d=1; %synaptic depre variable; see mx_wnNoReflBndry.c
sigp=[0.2;0.3]; %[sigE;sigI], noise strength of E & I cells

%---specify hetereogeneity vectors---

% Now included in parameter file
%newThres=ones(Ne+Ni,1);  %vector of thresholds; change to make heterog
qd_preF=ones(Ne,1);       %vector of synapt var, only for E cells; change to make heterog

%next 5 lines must match c-file mx_wnNoReflBndry.c!
T_win=[5; 50; 100]; %in ms

tic
    [nuE,nuI,mn_E,mn_I,var_E,var_I,icov_ee,icov_ie,snuE,snuI,sFFe,sFFi,sRhoEE,...
        sRhoIE,smn_E,smn_I,svarE,svarI,scovEE,scovIE] ...
        = mx_wnNoReflBndry(W_ei,W_ie,W_ee,W_ii,g_vec,id1_rie,id2_rie,id1_ree,id2_ree,qd_preF,newThres,a_d,sigp);
toc

dts=.001; %matches sample rate in mx_wnNoReflBndry.c
Lts=size(nuE,2);
t=(dts:dts: Lts*dts)'; %time MATCHES mx_Mn

%save various stats
save d_tt_asynch t nuE nuI mn_* var_* icov_* T_win snu* smn_* svar* scov* sFF* sRho* M_* sigp