#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mex.h"

/* Box-Muller; faster than using sin,cos */
 void z_randn(double *normal_rv, double *nrm_rv2)
 {
	 double u, v, s=2.0, z, z2;
	 while(s>=1 || s==0){
		 u=(double)(rand())/RAND_MAX*2-1;
		 v=(double)(rand())/RAND_MAX*2-1;
		 s=u*u+v*v;
	 }
	 z=sqrt(-2.0*log(s)/s)*u;
	 z2=z/u*v;
	 
	 *normal_rv=z;
	 *nrm_rv2=z2;
 }

void mexFunction(int nlhs, mxArray *plhs[ ],int nrhs, const mxArray *prhs[ ]) 
{
/* 
Like mx_Mwn.c BUT does not have reflecting boundary for either E/I, AND no sqrt(taum) in the white noise term
 
[nuE,nuI,mn_E,mn_I,var_E,var_I,icov_ee,icov_ie,snuE,snuI,sFFe,sFFi,sRhoEE,sRhoIE,smn_E,smn_I,svarE,svarI,scovEE,scovIE]=
 mx_wnNoReflBndry(W_ei,W_ie,W_ee,W_ii,g_vec,id1_rie,id2_rie,id1_ree,id2_ree,v_preF,Thres,a_d,[sigE;sigI]);
called in Matlab script
must compile first: >> mex mx_wnNoReflBndry.c
ALL TIMES IN MILLISECONDS
NO Synaptic failure (but track dyn probTrans), E/I prefactor, het. thres.
 [from as mx_Fail.c, (sigE/I, trial-to-trial stats), but with het. thresh]
 No background corr, but can add-in (see corr var
*/  

/*  Check for proper number of arguments */
if (nrhs !=  13) {
	mexErrMsgTxt("13 input arguments required.");
} else if (nlhs > 20) {
	mexErrMsgTxt("Too many output arguments.");
}

int Ni, Ne, szRee, szRie, i, j, k, ind1, N=100000, nstd=10, numStr=10000;
double taum=0.02, tRef=0.002, t_re=0.001, t_de=0.005, t_ri=0.002, t_di=0.01, Esyn=6.5, Isyn=-0.5; /* intrins neuro-params */
double ampE=1.0, ampI=2.0, sigI, sigE, sqtn; /* intrins neuro-params */
double tau_r=0.01, a_d, rndUnif; /* each event decr prob trans by x*a_d[input] IF successfully transmit; tau_r recovery to 1 */

int l_wei, l_wie, l_wee, l_wii;
double *Wei, *Wie, *Wee, *Wii, *g_vec, *qd_preF, *Thres, *a_dp, *sigp; /* passed-in */
double g_ei, g_ie, g_ee, g_ii, Gei, Gie, Gee, Gii; /* (some) passed-in */
double *id1_ried, *id2_ried, *id1_reed, *id2_reed;	/* passed-in */
	
double corr=0, c1, cc, etac, randc1, randc2=0.0, dt=0.00001, dts=0.001; /* used in for-loop; etac is common-noise */
int Lt=100000, Lts, spct, nrnSpace; /* 1 s for each realiz */

/* correl window params (ms); hard-coding in; COULD pass these pieces in as argument(s) */
int nmT=3, totWn=0, indx_wn, subIndx;
int numWins_rlz[nmT];
numWins_rlz[0]=200; /* 1/5ms=200 */
numWins_rlz[1]=20; /* 1/50ms=20 */
numWins_rlz[2]=10; /* 1/100ms=10 */
for (k=0; k<nmT; k++) {
    totWn+=numWins_rlz[k];
}
   
/* outputs */
double denom_eei=0;
double *nuE, *nuI, *mn_E, *mn_I, *var_E, *var_I, *icov_ee, *icov_ie, *snuE, *snuI, *sFFe,* sFFi, *sRhoEE, *sRhoIE, *smn_E, *smn_I, *svarE, *svarI, *scovEE, *scovIE;
	
/* Using mxGetScalar to retreive input arguments */
Wei=mxGetPr(prhs[0]);		  /* Ne x Ni vector */
Wie=mxGetPr(prhs[1]);		  /* Ni x Ne vector */
Wee=mxGetPr(prhs[2]);
Wii=mxGetPr(prhs[3]);
g_vec=mxGetPr(prhs[4]);
id1_ried=mxGetPr(prhs[5]);    /* vector szRie x 1, I index in I-E corr; CONVERT all 4 into ints in initializ */
id2_ried=mxGetPr(prhs[6]);    /* vector szRie x 1, E index in I-E corr */
id1_reed=mxGetPr(prhs[7]);    /* vector szRee x 1, E1 index in E-E corr */
id2_reed=mxGetPr(prhs[8]);    /* vector szRee x 1, E2 index in E-E corr */
qd_preF=mxGetPr(prhs[9]);
Thres=mxGetPr(prhs[10]);
a_dp=mxGetPr(prhs[11]);
sigp=mxGetPr(prhs[12]);

Ne=mxGetM(prhs[0]);    /* # rows in connect matrix */
Ni=mxGetM(prhs[1]);
l_wei=mxGetN(prhs[0]); /* # cols in conn matrix; smaller than Ne and Ni to save time */
l_wie=mxGetN(prhs[1]);
l_wee=mxGetN(prhs[2]);
l_wii=mxGetN(prhs[3]);	
szRie=mxGetM(prhs[5]); /* #rows in id1_ried */
szRee=mxGetM(prhs[7]);
g_ei=g_vec[0];
g_ie=g_vec[1];
g_ee=g_vec[2];
g_ii=g_vec[3];
a_d=a_dp[0];
sigE=sigp[0]; /*sigE is 1st entry, sigI is 2nd entry*/
sigI=sigp[1];
	
spct=(int)(dts/dt); /* save spikes every spct-time steps */
Lts=Lt/spct;   /* only sample ms; !Lts is total time (ms) */
	
/* number of time steps before out of refractory */
nrnSpace=(int)(tRef/dt);
	
/* OUTPUTs ... */
plhs[0]=mxCreateDoubleMatrix( Ne, Lts, mxREAL); /* only mxREAL part */
nuE = mxGetPr(plhs[0]);
plhs[1]=mxCreateDoubleMatrix( Ni, Lts, mxREAL); /* only mxREAL part */
nuI = mxGetPr(plhs[1]);
plhs[2]=mxCreateDoubleMatrix( Ne, nmT, mxREAL); /* only mxREAL part */
mn_E = mxGetPr(plhs[2]);
plhs[3]=mxCreateDoubleMatrix( Ni, nmT, mxREAL); /* only mxREAL part */
mn_I = mxGetPr(plhs[3]);
plhs[4]=mxCreateDoubleMatrix( Ne, nmT, mxREAL); /* only mxREAL part */
var_E = mxGetPr(plhs[4]);
plhs[5]=mxCreateDoubleMatrix( Ni, nmT, mxREAL); /* only mxREAL part */
var_I = mxGetPr(plhs[5]);
plhs[6]=mxCreateDoubleMatrix( szRee, nmT, mxREAL); /* only mxREAL part */
icov_ee = mxGetPr(plhs[6]);
plhs[7]=mxCreateDoubleMatrix( szRie, nmT, mxREAL); /* only mxREAL part */
icov_ie = mxGetPr(plhs[7]);
plhs[8]=mxCreateDoubleMatrix( Ne, 1, mxREAL); /* only mxREAL part */
snuE = mxGetPr(plhs[8]);
plhs[9]=mxCreateDoubleMatrix( Ni, 1, mxREAL); /* only mxREAL part */
snuI = mxGetPr(plhs[9]);
plhs[10]=mxCreateDoubleMatrix( Ne, nmT, mxREAL); /* only mxREAL part */
sFFe = mxGetPr(plhs[10]);
plhs[11]=mxCreateDoubleMatrix( Ni, nmT, mxREAL); /* only mxREAL part */
sFFi = mxGetPr(plhs[11]);
plhs[12]=mxCreateDoubleMatrix( szRee, nmT, mxREAL); /* only mxREAL part */
sRhoEE = mxGetPr(plhs[12]);
plhs[13]=mxCreateDoubleMatrix( szRie, nmT, mxREAL); /* only mxREAL part */
sRhoIE = mxGetPr(plhs[13]);
plhs[14]=mxCreateDoubleMatrix( Ne, nmT, mxREAL); /* only mxREAL part */
smn_E = mxGetPr(plhs[14]);
plhs[15]=mxCreateDoubleMatrix( Ni, nmT, mxREAL); /* only mxREAL part */
smn_I = mxGetPr(plhs[15]);
plhs[16]=mxCreateDoubleMatrix( Ne, nmT, mxREAL); /* only mxREAL part */
svarE = mxGetPr(plhs[16]);
plhs[17]=mxCreateDoubleMatrix( Ni, nmT, mxREAL); /* only mxREAL part */
svarI = mxGetPr(plhs[17]);
plhs[18]=mxCreateDoubleMatrix( szRee, nmT, mxREAL); /* only mxREAL part */
scovEE = mxGetPr(plhs[18]);
plhs[19]=mxCreateDoubleMatrix( szRie, nmT, mxREAL); /* only mxREAL part */
scovIE = mxGetPr(plhs[19]);

/* store spkCnt in vector/array for each window size */
/* correl and counts for running sum */
double spkC_fs[Ni*totWn];  /* spike counts all FS(I) realiz; (Ni x totWn ) */
double spkC_rs[Ne*totWn];  /* spike counts all RS(E) realiz; (Ne x totWn ) */
double muRS1[szRee*nmT]; /* mean of spike counts for RS-RS */
double muRS2[szRee*nmT]; /* mean of spike counts for RS-RS */
double muFR[szRie*nmT];  /* mean of spike counts for FS-RS (FS) */
double muRF[szRie*nmT];  /* mean of spike counts for FS-RS (RS) */
/* extra vars for std error bar calc */
double snuEtmp[Ne];     /* tmp storage */
double snuItmp[Ni];     /* tmp storage */
double Emu_e1[szRee*nmT]; /* tmp storage */
double Emu_e2[szRee*nmT]; /* tmp storage */
double Emu_fr[szRie*nmT];  /* tmp storage */
double Emu_rf[szRie*nmT];  /* tmp storage */
double scovEEtmp[szRee*nmT]; /* tmp storage */
double scovIEtmp[szRie*nmT]; /* tmp storage */
double smn_Itmp[Ni*nmT]; /* tmp storage */
double svarItmp[Ni*nmT]; /* tmp storage */
double smn_Etmp[Ne*nmT]; /* tmp storage */
double svarEtmp[Ne*nmT]; /* tmp storage */

    
/* used in for-loop; ODEs */
double vlt_I[Ni];    /* voltage */
double synFS[Ni];  /* synapses */
double asynFS[Ni]; /* aux syn */
double vlt_E[Ne];     
double synRS[Ne];
double asynRS[Ne];
double etaV[Ni+Ne]; /* indiv noise */
//double probTrans[Ni+Ne];
int TmRSspk[Ne]; /* for ref period, time last spike */
int TmFSspk[Ni];
int id1_ree[szRee];
int id2_ree[szRee];
int id1_rie[szRie];
int id2_rie[szRie];
int W_ei[Ne*l_wei];
int W_ie[Ni*l_wie];
int W_ee[Ne*l_wee];
int W_ii[Ni*l_wii];
	
/*  initialization */
srand ( time(NULL) ); /* seeing random # gen. */
sqtn=sqrt(1/dt); //here, there is NO sqrt(taum) unlike the Richardson theory
c1 = sqrt((1-corr));
cc = sqrt((corr));
for (i=0; i<Lts; i++) {
    for (j=0; j<Ni; j++) {
        nuI[i*Ni+j]=0.0;
        if (i==0) {
            snuI[j]=0.0;
        }
    }
}
for (i=0; i<Lts; i++) {
    for (j=0; j<Ne; j++) {
        nuE[i*Ne+j]=0.0;
        if (i==0) {
            snuE[j]=0.0;
        }
    }
}
/* more initialization; set corr pieces to 0 */
for (j=0; j<nmT; j++) {
	for (k=0; k<szRee; k++) {
		muRS1[j*szRee+k]=0.0; 
		muRS2[j*szRee+k]=0.0;
		icov_ee[j*szRee+k]=0.0;
        scovEE[j*szRee+k]=0.0;
        sRhoEE[j*szRee+k]=0.0;
        if (j==0) { /* Only need to do it once */
            id1_ree[k]=(int)id1_reed[k];
            id2_ree[k]=(int)id2_reed[k];
        }
	}
	for (k=0; k<szRie; k++) {
		muFR[j*szRie+k]=0.0; 
		muRF[j*szRie+k]=0.0;
		icov_ie[j*szRie+k]=0.0;
        scovIE[j*szRie+k]=0.0;
        sRhoIE[j*szRie+k]=0.0;
        if (j==0) { /* Only need to do it once */
            id1_rie[k]=(int)id1_ried[k];
            id2_rie[k]=(int)id2_ried[k];
        }
	}
    for (k=0; k<Ni; k++) {
        mn_I[j*Ni+k]=0.0;
        var_I[j*Ni+k]=0.0;
        smn_I[j*Ni+k]=0.0;
        svarI[j*Ni+k]=0.0;
        sFFi[j*Ni+k]=0.0;
    }
    for (k=0; k<Ne; k++) {
        mn_E[j*Ne+k]=0.0;
        var_E[j*Ne+k]=0.0;
        smn_E[j*Ne+k]=0.0;
        svarE[j*Ne+k]=0.0;
        sFFe[j*Ne+k]=0.0;
    }
}
    
for (j=0; j<Ne; j++) {
	for (k=0; k<l_wei; k++)
		W_ei[k*Ne+j]=(int)Wei[k*Ne+j];
	for (k=0; k<l_wee; k++)
		W_ee[k*Ne+j]=(int)Wee[k*Ne+j];
}
for (j=0; j<Ni; j++) {
	for (k=0; k<l_wie; k++)
		W_ie[k*Ni+j]=(int)Wie[k*Ni+j];
	for (k=0; k<l_wii; k++)
		W_ii[k*Ni+j]=(int)Wii[k*Ni+j];
}
	
srand ( time(NULL) ); /* seeing random # gen. */

/* set for FIRST run only; then Initi Cond. determined from t(end) of previous run */
for (j=0; j<(Ni+Ne); j++) {
	etaV[j] = ((double)(rand())/RAND_MAX-0.5)*2; /* rand unif i.c. [-1,1] */
//    probTrans[j]=1;
	if(j<Ni){
		vlt_I[j]=(double)(rand())/RAND_MAX; /* rand unif i.c. */
		synFS[j]=0.0;
		asynFS[j]=0.0;
	}
	else{
		vlt_E[j-Ni]=(double)(rand())/RAND_MAX; /* rand unif i.c. */
		synRS[j-Ni]=0.0;
		asynRS[j-Ni]=0.0;
	}
}
etac=((double)(rand())/RAND_MAX-0.5)*2;
		

/* --- Run it once to get rid of transients -- */
    for (j=0; j<Lt; j++){
        
        for (k=0; k<Ni; k++){
            if((j-TmFSspk[k]) >= nrnSpace ){ /* v change only if not in refrac */
                Gie=0.0; /* recalc */
                ind1=0;
                while (ind1<l_wie && W_ie[ind1*Ni+k]!=-1) {
                    Gie+=(g_ie*synRS[W_ie[ind1*Ni+k]]);
                    ind1++;
                }
                Gii=0.0; /* recalc */
                ind1=0;
                while (ind1<l_wii && W_ii[ind1*Ni+k]!=-1) {
                    Gii+=(g_ii*synFS[W_ii[ind1*Ni+k]]);
                    ind1++;
                }
                vlt_I[k]=vlt_I[k]+dt/taum*(-vlt_I[k]-Gie*(vlt_I[k]-Esyn)-Gii*(vlt_I[k]-Isyn)+sigI*sqtn*(c1*etaV[k] +cc*etac ));
                //if (vlt_I[k]<Isyn) { /* lower barrier */
                //    vlt_I[k]=2*Isyn-vlt_I[k];
                //}
            }
            synFS[k]=synFS[k]+dt/t_di*(-synFS[k]+asynFS[k]);
            asynFS[k]=asynFS[k]+dt/t_ri*(-asynFS[k]);
            //probTrans[k]=probTrans[k]+dt/tau_r*(1-probTrans[k]);
            if(k%2 == 0){
                z_randn(&randc1,&randc2); /*k starts at 0, so randc2 will be assigned */
                etaV[k]=randc1;
            }
            else
                etaV[k]=randc2;
            /* spiked */
            if(vlt_I[k]>=Thres[k]){
                asynFS[k]+=ampI; /* update synapse */
                vlt_I[k]=0.0;		 /* reset voltage */
                TmFSspk[k]=j;
            }
        }
        for (k=0; k<Ne; k++){
            if((j-TmRSspk[k]) >= nrnSpace ){ /* v change only if not in refrac */
                Gei=0.0; /* recalc */
                ind1=0;
                while (ind1<l_wei && W_ei[ind1*Ne+k]!=-1) {
                    Gei+=(qd_preF[k]*g_ei*synFS[W_ei[ind1*Ne+k]]);
                    ind1++;
                }
                Gee=0.0; /* recalc */
                ind1=0;
                while (ind1<l_wee && W_ee[ind1*Ne+k]!=-1) {
                    Gee+=(qd_preF[k]*g_ee*synRS[W_ee[ind1*Ne+k]]);
                    ind1++;
                }
                vlt_E[k]=vlt_E[k]+dt/taum*(-vlt_E[k]-Gei*(vlt_E[k]-Isyn)-Gee*(vlt_E[k]-Esyn)+sigE*sqtn*(c1*etaV[Ni+k] +cc*etac ));
                //if (vlt_E[k]<Isyn) { /* lower barrier */
                //    vlt_E[k]=2*Isyn-vlt_E[k];
                //}
            }
            synRS[k]=synRS[k]+dt/t_de*(-synRS[k]+asynRS[k]);
            asynRS[k]=asynRS[k]+dt/t_re*(-asynRS[k]);
            //probTrans[Ni+k]=probTrans[Ni+k]+dt/tau_r*(1-probTrans[Ni+k]);
            if(k%2 == 0){
                z_randn(&randc1,&randc2); /*k starts at 0, so randc2 will be assigned */
                etaV[Ni+k]=randc1;
            }
            else
                etaV[Ni+k]=randc2;
            /* spiked */
            if(vlt_E[k]>=Thres[Ni+k]){
                asynRS[k]+=ampE; /* update synapse */
                vlt_E[k]=0.0;		 /* reset voltage */
                TmRSspk[k]=j;
            }
        }
        z_randn(&etac,&randc2); /* common noise */
    }/* ending j-loop (transient time) */

/*  MAIN....N realizations. */
for (i=0; i<N; i++){
	for (j=0; j<(Ni+Ne); j++) { /* SET INITIAL CONDITIONS */
		if (j<Ni) {
            TmFSspk[j]=TmFSspk[j]-Lt;
            //    TmFSspk[j]=-nrnSpace;
			for (k=0; k<totWn; k++) {
				spkC_fs[j*totWn+k]=0.0; /* sloppy */
			}
		}
		else {
            TmRSspk[j-Ni]=TmRSspk[j-Ni]-Lt;
            //    TmRSspk[j-Ni]=-nrnSpace;
			for (k=0; k<totWn; k++) {
				spkC_rs[(j-Ni)*totWn+k]=0.0; /* sloppy */
			}
		}
	}
    /* only update every numStr; set all tmp pieces to 0 */
    if (i%numStr == 0) {
        for (j=0; j<nmT; j++) {
            for (k=0; k<szRee; k++) {
                Emu_e1[j*szRee+k]=0.0;
                Emu_e2[j*szRee+k]=0.0;
                scovEEtmp[j*szRee+k]=0.0;
            }
            for (k=0; k<szRie; k++) {
                Emu_fr[j*szRie+k]=0.0;
                Emu_rf[j*szRie+k]=0.0;
                scovIEtmp[j*szRie+k]=0.0;
            }
            for (k=0; k<Ni; k++) {
                if (j==0) {
                    snuItmp[k]=0.0; //only need it once, not nmT times
                }
                smn_Itmp[j*Ni+k]=0.0;
                svarItmp[j*Ni+k]=0.0;
            }
            for (k=0; k<Ne; k++) {
                if (j==0) {
                    snuEtmp[k]=0.0; //only need it once, not nmT times
                }
                smn_Etmp[j*Ne+k]=0.0;
                svarEtmp[j*Ne+k]=0.0;
            }
        }
    }
    
/* start of time-loop */
for (j=0; j<Lt; j++){
    
	for (k=0; k<Ni; k++){
		if((j-TmFSspk[k]) >= nrnSpace ){ /* v change only if not in refrac */
			Gie=0.0; /* recalc */
			ind1=0;
			while (ind1<l_wie && W_ie[ind1*Ni+k]!=-1) {
				Gie+=(g_ie*synRS[W_ie[ind1*Ni+k]]);
				ind1++;
			}
			Gii=0.0; /* recalc */
			ind1=0;
			while (ind1<l_wii && W_ii[ind1*Ni+k]!=-1) {
				Gii+=(g_ii*synFS[W_ii[ind1*Ni+k]]);
				ind1++;
			}
		
			vlt_I[k]=vlt_I[k]+dt/taum*(-vlt_I[k]-Gie*(vlt_I[k]-Esyn)-Gii*(vlt_I[k]-Isyn)+sigI*sqtn*(c1*etaV[k] +cc*etac ));
			//if (vlt_I[k]<Isyn) { /* lower barrier */
			//	vlt_I[k]=2*Isyn-vlt_I[k];
			//}
		}
		synFS[k]=synFS[k]+dt/t_di*(-synFS[k]+asynFS[k]);
		asynFS[k]=asynFS[k]+dt/t_ri*(-asynFS[k]);
		//probTrans[k]=probTrans[k]+dt/tau_r*(1-probTrans[k]);
        
		if(k%2 == 0){
			z_randn(&randc1,&randc2); /*k starts at 0, so randc2 will be assigned */
			etaV[k]=randc1;
		}
		else
			etaV[k]=randc2;
		
		/* spiked */
		if(vlt_I[k]>=Thres[k]){
            /*implement probTrans */
            //rndUnif=(double)(rand())/RAND_MAX;
            //if(rndUnif < probTrans[k]){
                asynFS[k]+=ampI; /* update synapse */
            //    probTrans[k]*=a_d; /* scale down probTrans */
            //}
			vlt_I[k]=0.0;		 /* reset voltage */
			nuI[j/spct*Ni+k]+=1; /* record spike; rely on integer division */
            snuItmp[k]+=1;
            
			for (ind1=0; ind1<nmT; ind1++) {/* add to window count */
                indx_wn=(int)(j*dt*numWins_rlz[ind1]);
                for (subIndx=ind1-1; subIndx>=0; subIndx--) {
                    indx_wn+=numWins_rlz[subIndx];
                }
				spkC_fs[indx_wn*Ni+k]+=1;
			}
			
			TmFSspk[k]=j;
		}

	}
	for (k=0; k<Ne; k++){		
		if((j-TmRSspk[k]) >= nrnSpace ){ /* v change only if not in refrac */
			Gei=0.0; /* recalc */
			ind1=0;
			while (ind1<l_wei && W_ei[ind1*Ne+k]!=-1) {
				Gei+=(qd_preF[k]*g_ei*synFS[W_ei[ind1*Ne+k]]);
				ind1++;
			}
			
			Gee=0.0; /* recalc */
			ind1=0;
			while (ind1<l_wee && W_ee[ind1*Ne+k]!=-1) {
				Gee+=(qd_preF[k]*g_ee*synRS[W_ee[ind1*Ne+k]]);
				ind1++;
			}
			
			vlt_E[k]=vlt_E[k]+dt/taum*(-vlt_E[k]-Gei*(vlt_E[k]-Isyn)-Gee*(vlt_E[k]-Esyn)+sigE*sqtn*(c1*etaV[Ni+k] +cc*etac ));
			//if (vlt_E[k]<Isyn) { /* lower barrier */
			//	vlt_E[k]=2*Isyn-vlt_E[k];
			//}
		}
		synRS[k]=synRS[k]+dt/t_de*(-synRS[k]+asynRS[k]);
		asynRS[k]=asynRS[k]+dt/t_re*(-asynRS[k]);
        //probTrans[Ni+k]=probTrans[Ni+k]+dt/tau_r*(1-probTrans[Ni+k]);
		
		if(k%2 == 0){
			z_randn(&randc1,&randc2); /*k starts at 0, so randc2 will be assigned */
			etaV[Ni+k]=randc1;
		}
		else
			etaV[Ni+k]=randc2;
		
		/* spiked */
		if(vlt_E[k]>=Thres[Ni+k]){
            /*implement probTrans */
            //rndUnif=(double)(rand())/RAND_MAX;
            //if(rndUnif < probTrans[Ni+k]){
                asynRS[k]+=ampE; /* update synapse */
            //    probTrans[Ni+k]*=a_d; /* scale down probTrans */
            //}
			vlt_E[k]=0.0;		 /* reset voltage */
			nuE[j/spct*Ne+k]+=1; /* record spike; rely on integer division */
            snuEtmp[k]+=1;
            
            for (ind1=0; ind1<nmT; ind1++) {/* add to window count */
                indx_wn=(int)(j*dt*numWins_rlz[ind1]);
                for (subIndx=ind1-1; subIndx>=0; subIndx--) {
                    indx_wn+=numWins_rlz[subIndx];
                }
				spkC_rs[indx_wn*Ne+k]+=1;
			}
			
			TmRSspk[k]=j;
		}

	}
    z_randn(&etac,&randc2); /* common noise */
}/* ending j-loop (time) */

	
/* UPDATE running sum so unbiased-estim of var,cov, etc */
for (j=0; j<nmT; j++) {
    //starting index, depending on j (nmT); to loop over all windows
    subIndx=0;
    for (k=j-1; k>=0; k--) {
        subIndx+=numWins_rlz[k];
    }
	for (k=0; k<szRee; k++) {
        for (ind1=subIndx; ind1<(subIndx+numWins_rlz[j]); ind1++) { //loop through all windows
            muRS1[j*szRee+k]+=spkC_rs[ind1*Ne+id1_ree[k]];
            muRS2[j*szRee+k]+=spkC_rs[ind1*Ne+id2_ree[k]];
            icov_ee[j*szRee+k]+=spkC_rs[ind1*Ne+id1_ree[k]]*spkC_rs[ind1*Ne+id2_ree[k]];
            Emu_e1[j*szRee+k]+=spkC_rs[ind1*Ne+id1_ree[k]];
            Emu_e2[j*szRee+k]+=spkC_rs[ind1*Ne+id2_ree[k]];
            scovEEtmp[j*szRee+k]+=spkC_rs[ind1*Ne+id1_ree[k]]*spkC_rs[ind1*Ne+id2_ree[k]];
        }
	}
	for (k=0; k<szRie; k++) {
        for (ind1=subIndx; ind1<(subIndx+numWins_rlz[j]); ind1++) { //loop through all windows
            muFR[j*szRie+k]+=spkC_fs[ind1*Ni+id1_rie[k]];
            muRF[j*szRie+k]+=spkC_rs[ind1*Ne+id2_rie[k]];
            icov_ie[j*szRie+k]+=spkC_fs[ind1*Ni+id1_rie[k]]*spkC_rs[ind1*Ne+id2_rie[k]];
            Emu_fr[j*szRie+k]+=spkC_fs[ind1*Ni+id1_rie[k]];
            Emu_rf[j*szRie+k]+=spkC_rs[ind1*Ne+id2_rie[k]];
            scovIEtmp[j*szRie+k]+=spkC_fs[ind1*Ni+id1_rie[k]]*spkC_rs[ind1*Ne+id2_rie[k]];
        }
	}
    for (k=0; k<(Ni+Ne); k++) {
        if (k<Ni) {
            for (ind1=subIndx; ind1<(subIndx+numWins_rlz[j]); ind1++) { //loop through all windows
                mn_I[j*Ni+k]+=spkC_fs[ind1*Ni+k];
                var_I[j*Ni+k]+=spkC_fs[ind1*Ni+k]*spkC_fs[ind1*Ni+k];
                smn_Itmp[j*Ni+k]+=spkC_fs[ind1*Ni+k];
                svarItmp[j*Ni+k]+=spkC_fs[ind1*Ni+k]*spkC_fs[ind1*Ni+k];
            }
        }
        else{
            for (ind1=subIndx; ind1<(subIndx+numWins_rlz[j]); ind1++) { //loop through all windows
                mn_E[j*Ne+k-Ni]+=spkC_rs[ind1*Ne+k-Ni];
                var_E[j*Ne+k-Ni]+=spkC_rs[ind1*Ne+k-Ni]*spkC_rs[ind1*Ne+k-Ni];
                smn_Etmp[j*Ne+k-Ni]+=spkC_rs[ind1*Ne+k-Ni];
                svarEtmp[j*Ne+k-Ni]+=spkC_rs[ind1*Ne+k-Ni]*spkC_rs[ind1*Ne+k-Ni];
            }
        }
    }
    /* Only update the running sum every numStr */
    if (i%numStr == numStr-1) {
        for (k=0; k<szRee; k++) {
            /* do normalization here for running sum of err bars */
            scovEE[j*szRee+k] += (scovEEtmp[j*szRee+k] - Emu_e1[j*szRee+k]*Emu_e2[j*szRee+k]/(numStr*numWins_rlz[j]))*(scovEEtmp[j*szRee+k] - Emu_e1[j*szRee+k]*Emu_e2[j*szRee+k]/(numStr*numWins_rlz[j]))/((numStr*numWins_rlz[j]-1)*(numStr*numWins_rlz[j]-1));
            if (svarEtmp[j*Ne+id1_ree[k]]>0 && svarEtmp[j*Ne+id2_ree[k]]>0) {
                //only execute if you have firing from both E-cells
                denom_eei=(svarEtmp[j*Ne+id1_ree[k]]-smn_Etmp[j*Ne+id1_ree[k]]*smn_Etmp[j*Ne+id1_ree[k]]/(numStr*numWins_rlz[j]))*(svarEtmp[j*Ne+id2_ree[k]]-smn_Etmp[j*Ne+id2_ree[k]]*smn_Etmp[j*Ne+id2_ree[k]]/(numStr*numWins_rlz[j]));
                sRhoEE[j*szRee+k] += (scovEEtmp[j*szRee+k] - Emu_e1[j*szRee+k]*Emu_e2[j*szRee+k]/(numStr*numWins_rlz[j]))*(scovEEtmp[j*szRee+k] - Emu_e1[j*szRee+k]*Emu_e2[j*szRee+k]/(numStr*numWins_rlz[j]))/denom_eei;
            }//else {sRhoEE[j*szRee+k]+=0;} //in case no firing; don't add anything
        }
        for (k=0; k<szRie; k++) {
            /* do normalization here for running sum of err bars */
            scovIE[j*szRie+k] += (scovIEtmp[j*szRie+k] - Emu_fr[j*szRie+k]*Emu_rf[j*szRie+k]/(numStr*numWins_rlz[j]))*(scovIEtmp[j*szRie+k] - Emu_fr[j*szRie+k]*Emu_rf[j*szRie+k]/(numStr*numWins_rlz[j]))/((numStr*numWins_rlz[j]-1)*(numStr*numWins_rlz[j]-1));
            if (svarItmp[j*Ni+id1_rie[k]]>0 && svarEtmp[j*Ne+id2_rie[k]]>0) {
                //only execute if you have firing from both E and I cells
                denom_eei=(svarItmp[j*Ni+id1_rie[k]] - smn_Itmp[j*Ni+id1_rie[k]]*smn_Itmp[j*Ni+id1_rie[k]]/(numStr*numWins_rlz[j]))*(svarEtmp[j*Ne+id2_rie[k]]-smn_Etmp[j*Ne+id2_rie[k]]*smn_Etmp[j*Ne+id2_rie[k]]/(numStr*numWins_rlz[j]));
                sRhoIE[j*szRie+k] += (scovIEtmp[j*szRie+k] - Emu_fr[j*szRie+k]*Emu_rf[j*szRie+k]/(numStr*numWins_rlz[j]))*(scovIEtmp[j*szRie+k] - Emu_fr[j*szRie+k]*Emu_rf[j*szRie+k]/(numStr*numWins_rlz[j]))/denom_eei;
            }//else {sRhoIE[j*szRie+k]+=0;} //in case no firing; don't add anything
        }
        for (k=0; k<(Ni+Ne); k++) {
            if (k<Ni) {
            /* do normalization here for running sum of err bars */
                smn_I[j*Ni+k] += smn_Itmp[j*Ni+k]*smn_Itmp[j*Ni+k]/((numStr*numWins_rlz[j])*(numStr*numWins_rlz[j]));
                svarI[j*Ni+k] += ( svarItmp[j*Ni+k] - smn_Itmp[j*Ni+k]*smn_Itmp[j*Ni+k]/(numStr*numWins_rlz[j]) )*( svarItmp[j*Ni+k] - smn_Itmp[j*Ni+k]*smn_Itmp[j*Ni+k]/(numStr*numWins_rlz[j]) )/((numStr*numWins_rlz[j]-1)*(numStr*numWins_rlz[j]-1));
                if (smn_Itmp[j*Ni+k]>0) { //only execute if you have firing from I-cell
                    sFFi[j*Ni+k] += ( svarItmp[j*Ni+k] - smn_Itmp[j*Ni+k]*smn_Itmp[j*Ni+k]/(numStr*numWins_rlz[j]) )*( svarItmp[j*Ni+k] - smn_Itmp[j*Ni+k]*smn_Itmp[j*Ni+k]/(numStr*numWins_rlz[j]) )/
                    ( smn_Itmp[j*Ni+k]*smn_Itmp[j*Ni+k] ); //denom N are technically off by -1, but large N doesn't matter
                }
            }
            else{
            /* do normalization here for running sum of err bars */
                smn_E[j*Ne+k-Ni] += smn_Etmp[j*Ne+k-Ni]*smn_Etmp[j*Ne+k-Ni]/((numStr*numWins_rlz[j])*(numStr*numWins_rlz[j]));
                svarE[j*Ne+k-Ni] += (svarEtmp[j*Ne+k-Ni]-smn_Etmp[j*Ne+k-Ni]*smn_Etmp[j*Ne+k-Ni]/(numStr*numWins_rlz[j]))*(svarEtmp[j*Ne+k-Ni]-smn_Etmp[j*Ne+k-Ni]*smn_Etmp[j*Ne+k-Ni]/(numStr*numWins_rlz[j]))/((numStr*numWins_rlz[j]-1)*(numStr*numWins_rlz[j]-1));
                if (smn_Etmp[j*Ne+k-Ni]>0) { //only execute if you have firing from E-cell
                    sFFe[j*Ne+k-Ni] += (svarEtmp[j*Ne+k-Ni]-smn_Etmp[j*Ne+k-Ni]*smn_Etmp[j*Ne+k-Ni]/(numStr*numWins_rlz[j]))*(svarEtmp[j*Ne+k-Ni]-smn_Etmp[j*Ne+k-Ni]*smn_Etmp[j*Ne+k-Ni]/(numStr*numWins_rlz[j]))/( smn_Etmp[j*Ne+k-Ni]*smn_Etmp[j*Ne+k-Ni] ); //denom N are technically off by -1, but large N doesn't matter
                }
            }
        }
        if (j==0) { /* only do it once, no loop over nmT */
            for (k=0; k<(Ni+Ne); k++) {
                if (k<Ni) {
                    snuI[k] += snuItmp[k]*snuItmp[k]/((Lts*numStr*dt*spct)*(Lts*numStr*dt*spct));
                }
                else{
                    snuE[k-Ni] += snuEtmp[k-Ni]*snuEtmp[k-Ni]/((Lts*numStr*dt*spct)*(Lts*numStr*dt*spct));
                }
            }
        }
    } /*end of numStr updates */
    
} /*end of j=0 to nmT */
	
} /* ending i-loop (realizations) */

/*  Normalization by N (mu,v,cv).. unraveling var,cov, etc */
for (j=0; j<nmT; j++) {
	for (k=0; k<szRee; k++) {
		muRS1[j*szRee+k]/=(N*numWins_rlz[j]);
		muRS2[j*szRee+k]/=(N*numWins_rlz[j]);
		icov_ee[j*szRee+k] = ( icov_ee[j*szRee+k] - (N*numWins_rlz[j])*muRS1[j*szRee+k]*muRS2[j*szRee+k] )/((N*numWins_rlz[j])-1);
	}
	for (k=0; k<szRie; k++) {
		muFR[j*szRie+k]/=(N*numWins_rlz[j]);
		muRF[j*szRie+k]/=(N*numWins_rlz[j]);
		icov_ie[j*szRie+k] = ( icov_ie[j*szRie+k] - (N*numWins_rlz[j])*muFR[j*szRie+k]*muRF[j*szRie+k] )/((N*numWins_rlz[j])-1);
	}
    for (k=0; k<(Ni+Ne); k++) {
        if (k<Ni) {
            mn_I[j*Ni+k]/=(N*numWins_rlz[j]);
            var_I[j*Ni+k] = ( var_I[j*Ni+k] - (N*numWins_rlz[j])*mn_I[j*Ni+k]*mn_I[j*Ni+k] )/((N*numWins_rlz[j])-1);
        }
        else{
            mn_E[j*Ne+k-Ni]/=(N*numWins_rlz[j]);
            var_E[j*Ne+k-Ni] = ( var_E[j*Ne+k-Ni] - (N*numWins_rlz[j])*mn_E[j*Ne+k-Ni]*mn_E[j*Ne+k-Ni] )/((N*numWins_rlz[j])-1);
        }
	}
}
for (j=0; j<(Lts); j++) { /* normalize now so don't have to keep running sum [not often spiking] */
	for (k=0; k<Ni; k++) {
		nuI[j*Ni+k]=nuI[j*Ni+k]/(N*dt*spct);
	}
	for (k=0; k<Ne; k++) {
		nuE[j*Ne+k]=nuE[j*Ne+k]/(N*dt*spct);
	}
}

/* Unravel the std err Bars defn */
    for (j=0; j<nmT; j++) {
        for (k=0; k<szRee; k++) {
            scovEE[j*szRee+k] = ( scovEE[j*szRee+k]-nstd*icov_ee[j*szRee+k]*icov_ee[j*szRee+k] )/(nstd-1);
            if (var_E[j*Ne+id1_ree[k]]>0 && var_E[j*Ne+id2_ree[k]]>0) {
                //only execute if you have firing from both E-cells
                sRhoEE[j*szRee+k] = ( sRhoEE[j*szRee+k]-nstd*icov_ee[j*szRee+k]*icov_ee[j*szRee+k]/(var_E[j*Ne+id1_ree[k]]*var_E[j*Ne+id2_ree[k]]) )/(nstd-1);
            }
            else { //in case no firing, set to 0
                sRhoEE[j*szRee+k]=0;
            }
        }
        for (k=0; k<szRie; k++) {
            scovIE[j*szRie+k] = ( scovIE[j*szRie+k]-nstd*icov_ie[j*szRie+k]*icov_ie[j*szRie+k] )/(nstd-1);
            if (var_I[j*Ni+id1_rie[k]]>0 && var_E[j*Ne+id2_rie[k]]>0) {
                //only execute if you have firing from both E and I cells
                sRhoIE[j*szRie+k] = ( sRhoIE[j*szRie+k]-nstd*icov_ie[j*szRie+k]*icov_ie[j*szRie+k]/(var_I[j*Ni+id1_rie[k]]*var_E[j*Ne+id2_rie[k]]) )/(nstd-1);
            }
            else { //in case no firing, set to 0
                sRhoIE[j*szRie+k]=0;
            }
        }
        for (k=0; k<(Ni+Ne); k++) {
            if (k<Ni) {
                smn_I[j*Ni+k] = ( smn_I[j*Ni+k] - nstd*mn_I[j*Ni+k]*mn_I[j*Ni+k] )/(nstd-1);
                svarI[j*Ni+k] = ( svarI[j*Ni+k] - nstd*var_I[j*Ni+k]*var_I[j*Ni+k] )/(nstd-1);
                if (mn_I[j*Ni+k]>0) {
                    //only execute if you have firing from I cells
                    sFFi[j*Ni+k] = ( sFFi[j*Ni+k]-nstd*var_I[j*Ni+k]*var_I[j*Ni+k]/(mn_I[j*Ni+k]*mn_I[j*Ni+k]) )/(nstd-1);
                }
                else { //in case no firing, set to 0
                    sFFi[j*Ni+k]=0;
                }
                
            }
            else{
                smn_E[j*Ne+k-Ni] = ( smn_E[j*Ne+k-Ni] - nstd*mn_E[j*Ne+k-Ni]*mn_E[j*Ne+k-Ni])/(nstd-1);
                svarE[j*Ne+k-Ni] = ( svarE[j*Ne+k-Ni] - nstd*var_E[j*Ne+k-Ni]*var_E[j*Ne+k-Ni])/(nstd-1);
                if (mn_E[j*Ne+k-Ni]>0) {
                    //only execute if you have firing from E cells
                    sFFe[j*Ne+k-Ni] = ( sFFe[j*Ne+k-Ni]-nstd*var_E[j*Ne+k-Ni]*var_E[j*Ne+k-Ni]/(mn_E[j*Ne+k-Ni]*mn_E[j*Ne+k-Ni]) )/(nstd-1);
                }
                else { //in case no firing, set to 0
                    sFFe[j*Ne+k-Ni]=0;
                }
            }
        }
    }
    
    for (k=0; k<Ni; k++) {
        snuItmp[k]=0.0; /* store avg rate nuI in snuItmp */
        for (j=0; j<Lts; j++) {
            snuItmp[k] += nuI[j*Ni+k]/Lts;
        }
        snuI[k] = (snuI[k] - nstd*snuItmp[k]*snuItmp[k])/(nstd-1);
    }
    for (k=0; k<Ne; k++) {
        snuEtmp[k]=0.0; /* store avg rate nuE in snuEtmp */
        for (j=0; j<Lts; j++) {
            snuEtmp[k] += nuE[j*Ne+k]/Lts;
        }
        snuE[k] = (snuE[k] - nstd*snuEtmp[k]*snuEtmp[k])/(nstd-1);
    }
	
return;
                
}