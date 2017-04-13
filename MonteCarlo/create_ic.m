%create_ic.m: create parameter file (used by both MC and Lin Resp. codes)
%       for relatively small N; 
% 
% small N -> save ALL EE and EI corr coeffs are saved
%
% Copyright: A.K. Barreiro and C. Ly, 2017
%  Please cite: Barreiro, A.K. and C. Ly. "When do Correlations Increase with Firing 
%      Rates in Recurrent Networks?" PLoS Computational Biology, 2017.
% 
%

Ne=80; %total E cells
Ni=20;%total I cells
Ntot=Ne+Ni;

%----First create threshold vector---
e_ind=(1:Ne)';

percU_e=(0.05: 0.9/(Ne-1): 0.95)'; %equally spaced CDF values b/c N is small!
percU_i=(0.05: 0.9/(Ni-1): 0.95)'; %equally spaced; first Ni are Inhib
percU_a=[percU_i; percU_e];

% Variance for lognormal distribution of thresholds
% Values used in paper: 0.2 (heterogeneous) and 0 (homogeneous) 
sigq_max=0.2;
if (sigq_max > 0)
    Thres_t=logninv(percU_a,0,sigq_max);%exp(sp_t(j)*sigq_max*randn(Ni+Ne,1));
else % Homogeneous thresholds
    Thres_t=ones(Ntot,1);
end
Thres_new=Thres_t;

% --- now create random matrix coupling ---
fr_ItoE=0.35;    %fractions; connectiv & # of correl
fr_EtoI=0.2;
fr_ItoI=0.4;
fr_EtoE=0.4;


gei=10;   %I->E

gie = 8;   %E->I (Str Asyn)
%gie=5;    %E->I (Asyn)

gee=9; %E->E (Str Asyn)
%gee=0.5; %E->E (Asyn)

gii=5;    %I->I

Nm_ei=round(Ni*fr_ItoE); %avg number of I inputs to E
Nm_ie=round(fr_EtoI*Ne);
Nm_ee=round(fr_EtoE*Ne);
Nm_ii=round(Ni*fr_ItoI); %avg number of I inputs to I

%deterministic number of connections
wEI=ones(Ne,1)*Nm_ei;%poissrnd(Nm_ei,Ne,1);
wIE=ones(Ni,1)*Nm_ie;%poissrnd(Nm_ie,Ni,1);
wEE=ones(Ne,1)*Nm_ee;%poissrnd(Nm_ee,Ne,1);
wII=ones(Ni,1)*Nm_ii;%poissrnd(Nm_ii,Ni,1);
%random number of conn; use int. Gaussian std 3
% wEI=Nm_ei+round(3*randn(Ne,1));
% wIE=Nm_ie+round(3*randn(Ni,1));
% wEE=Nm_ee+round(3*randn(Ne,1));
% wII=Nm_ii+round(3*randn(Ni,1));
% wEI=poissrnd(Nm_ei,Ne,1);
% wIE=poissrnd(Nm_ie,Ni,1);
% wEE=poissrnd(Nm_ee,Ne,1);
% wII=poissrnd(Nm_ii,Ni,1);

J_ei=zeros(Ne,Ni); %weight matrix
for j=1:Ne
   %ind=randi(Ni,1,wEI(j));  %some repeats; less than wEI
   ind=randperm(Ni,wEI(j))';
   J_ei(j,ind)=1;
end
%J_ei=ones(Ne,Ni); %remove comment to make it all-to-all
%J_ei=gei*J_ei./Nm_ei; %J_ei=g*J_ei./Ni; %all-to-all

J_ie=zeros(Ni,Ne);
for j=1:Ni
   %ind=randi(Ne,1,wIE(j)); %overlap; less than wIE
   ind=randperm(Ne,wIE(j))';
   J_ie(j,ind)=1;
end
%J_ie=ones(Ni,Ne); %all-to-all
%J_ie=gie*J_ie./Nm_ie; %J_ie=gie*J_ie./Ne; %all-to-all

J_ee=zeros(Ne,Ne);
for j=1:Ne
   %ind=randi(Ne,1,wEE(j)); %overlap; less than wEE
   ind=randperm(Ne,wEE(j))';
   J_ee(j,ind)=1;
end
J_ee=J_ee-diag(diag(J_ee));
%J_ee=ones(Ne,Ne); %all-to-all
%J_ee=gee*J_ee./Nm_ee; %J_ee=gee*J_ee./Ne; %all-to-all

J_ii=zeros(Ni,Ni); %weight matrix
for j=1:Ni
   %ind=randi(Ni,1,wII(j)); %overlap; less than wII
   ind=randperm(Ni,wII(j))';
   J_ii(j,ind)=1;
end
J_ii=J_ii-diag(diag(J_ii));
%J_ii=ones(Ni,Ni); %all-to-all
%J_ii=gii*J_ii./Nm_ii; %J_ei=g*J_ei./Ni; %all-to-all

%--trimming so don't have such large connectivity martrices--
nm_cols=max(sum(J_ei,2));
W_ei=zeros(Ne,nm_cols);
nm_cols=max(sum(J_ii,2));
W_ii=zeros(Ni,nm_cols);
nm_cols=max(sum(J_ie,2));
W_ie=zeros(Ni,nm_cols);
nm_cols=max(sum(J_ee,2));
W_ee=zeros(Ne,nm_cols);
for j=1:Ni
   indx=find(J_ie(j,:));
   W_ie(j,1:length(indx))=indx;
   indx=find(J_ii(j,:));
   W_ii(j,1:length(indx))=indx;
end
for j=1:Ne
   indx=find(J_ei(j,:));
   W_ei(j,1:length(indx))=indx;
   indx=find(J_ee(j,:));
   W_ee(j,1:length(indx))=indx;
end
Nconn_ei=W_ei; %store Nconn_ei to keep track of actual number of connections
Nconn_ie=W_ie;
Nconn_ee=W_ee;
Nconn_ii=W_ii;
Nconn_ei(Nconn_ei~=0)=1;
Nconn_ie(Nconn_ie~=0)=1;
Nconn_ee(Nconn_ee~=0)=1;
Nconn_ii(Nconn_ii~=0)=1;
%in C: 0,1,..,N-1 and -1 is where stop in C for-loop
W_ei=W_ei-1;
W_ii=W_ii-1;
W_ie=W_ie-1;
W_ee=W_ee-1;

% Which EI correlation pairs to save?  (do not need to save them all)
%
%rho_EI pairs; go through all Ni systematically
id1_rie=(0:Ni-1)'; %index of I cells (between 0 and Ni-1; C index)
id2_rie=(0:Ne-1)'; %index of E cells (beetween 0 and Ne-1; C index)
id1_rie=repmat(id1_rie,Ne,1);
id2_rie=reshape(repmat(id2_rie',Ni,1),Ni*Ne,1);
szRie=length(id2_rie); %Ne*Ni; total number of rho_EI pairs

% Which EE correlation pairs to save?  (do not need to save them all)
%
%rho_EE pairs
mt_ee=[];
for j=1:(Ne-1)
    mt_ee=[mt_ee; (j-1)*ones(Ne-j,1) (j:Ne-1)'];
end
szRee=size(mt_ee,1); %Ne*(Ne-1)/2; total number of rho_EE pairs
id1_ree=mt_ee(:,1);
id2_ree=mt_ee(:,2);

gRaw_vec=[gei;gie;gee;gii];
%normalize syn-strength by fraction of connected
g_ei=gei/(Ni*fr_ItoE);
g_ie=gie/(Ne*fr_EtoI);
g_ee=gee/(Ne*fr_EtoE);
g_ii=gii/(Ni*fr_ItoI);

g_vec=[g_ei;g_ie;g_ee;g_ii];

save strAsyPs_het Ne Ni W_* Nconn_* g_vec gRaw_vec id1_* id2_* szRee szRie Thres_new