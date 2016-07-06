
from sys import argv
import camb
from camb import model, initialpower
from scipy.linalg import cho_factor, cho_solve, cholesky, inv
from math import pi, log
from numpy import inf, append, loadtxt, array, dot, isnan, identity, arange, exp, diag, hstack,delete,outer, zeros, shape
from matplotlib.pyplot import plot
from scipy.optimize import minimize
import os.path as osp
from numpy.random import multivariate_normal
import json
from itertools import takewhile

spectra = argv[1]
sim_start = int(argv[2])
sim_end = int(argv[3])
lmax = int(argv[4])

class cmb_model():
    
    def __init__(self,lmax = 4000):
        
        self.lmax = lmax
        self.pars = camb.CAMBparams()
        self.pars.set_for_lmax(self.lmax, lens_potential_accuracy=1)

    def __call__(self, fit_params):
        self.pars.set_cosmology(H0=None, cosmomc_theta=fit_params['theta']/100.,
                           ombh2=fit_params['ombh2'],
                           omch2=fit_params['ommh2']-fit_params['ombh2'],
                           tau=fit_params['tau']
                           )
                                      
        self.pars.InitPower.set_params(As=fit_params['clamp']*1e-9*exp(2*fit_params['tau']),
                                  ns=fit_params['ns'])
                                                                           
        powers = camb.get_results(self.pars).get_cmb_power_spectra(self.pars)

        return powers['lensed_scalar'][:self.lmax,0]*(1e6)**2 * 2.725**2
    
    
class like():
    
    def __init__(self, spectra, sim_num,lmax = 2500, use_data = False, whiten = True, **kwargs):
        
        self.freqs = map(int,[spectra[:3],spectra[4:]])
        if spectra == '150x150':
            sim_path = 'sims/150x150sims/'
            self.param_cov = 'params_for_whitening_150x150.txt'
            cal = 1.0091**2
        elif spectra ==  '150x143':
            sim_path = 'sims/150x143sims/'
            self.param_cov = 'params_for_whitening_150x143.txt'
            cal = 1.0091
        elif spectra == '143x143':
            sim_path = 'sims/143x143sims/'
            self.param_cov = 'params_for_whitening_143x143.txt'
            cal = 1
        
        else:
            print 'needs to be 150x150, 150x143, or 143x143'
       
        self.param_names = ['theta','ombh2','ommh2','clamp','ns','tau','Asz','Aps','Acib']

        if not use_data: 
            self.spectra = loadtxt(sim_path+'sims400/bandpower_sim_'+sim_num+'.txt')[:,-1]
        else:
            self.spectra = loadtxt(sim_path+'/bandpower.txt')[:,-1]
            
        self.sigma = loadtxt(sim_path+'covariance.txt')
        
        self.windows = array([loadtxt(sim_path+'window/window_%i'%i)[:,1] for i in range(37)])
        self.windowrange = (lambda x: slice(int(min(x)),int(max(x)+1)))(loadtxt(sim_path+'window/window_1')[:,0])
        self.lmax = int(self.windowrange.stop)
        
        if lmax is not None:
            bmax = sum(1 for _ in takewhile(lambda x: x<lmax, [3251 - sum(1 for _ in takewhile(lambda x: abs(x)<.001,reversed(w)) ) for w in self.windows]))
        else: bmax = 37
        
        self.spectra = self.spectra[:bmax]*cal
        self.sigma = self.sigma[:bmax,:bmax]*cal**2
        self.windows = self.windows[:bmax] 
              
        if whiten:
            self.cho_param_cov = cholesky(loadtxt(sim_path+self.param_cov)).T
        else:
            self.cho_param_cov = identity(7)
                    
        self.beam_corr = beam_errs(freqs=self.freqs)[self.windowrange,self.windowrange]
        self.fgs = fgs_model(self.freqs)
        self.tSZ_dep = self.fgs.tsz_dep
        self.cib_dep = self.fgs.cib_dep
        self.sz_prior = 5.5
        self.ps_prior = 19.3
        self.cib_prior = 5.0
        self.cmb_model = cmb_model(lmax = self.lmax)
        
    def cl_model(self, fit_params):
        
        spectra_theory = self.cmb_model(fit_params)[self.windowrange]
        fgs = self.fgs(fit_params,self.lmax)[self.windowrange]
        model = (spectra_theory+fgs)
        return model
      
    def fgs_priors(self, fit_params):
        
        return (fit_params['Asz']-self.sz_prior)**2./(2.*3.**2.)+(fit_params['Aps']-self.ps_prior)**2./(2.*3.5**2.)+\
                (fit_params['Acib']-self.cib_prior)**2./(2.*2.5**2.)
            
            
    def __call__(self, fit_params):
        
        fit_params = dict(zip(self.param_names,dot(self.cho_param_cov,fit_params)))
        try:
            model =  self.cl_model(fit_params)
        except ValueError:
            return inf
        dcl = self.spectra - dot(self.windows,model)
        if (self.freqs[0]== 150):
            model_matrix = outer(model,model)
            beam_cov = dot(self.windows,dot(self.beam_corr*model_matrix,self.windows.T))
        else:
            beam_cov = 0
        cho_cov = cho_factor(self.sigma+beam_cov)
        
        lnl = dot(dcl,cho_solve(cho_cov, dcl))/2   + \
        (fit_params['tau']-0.066)**2/(2*0.02**2) + \
        self.fgs_priors(fit_params)
       
        return lnl
    
    
def beam_errs(freqs=[150,150],lmax=4000,**kwargs): 
    names = ['dc','alpha','wobble','xtalk','outer','inner','venus'] 
    years = ['2008','2009','2010','2011']
    year_weights = {'2008': 0.065919966,'2009': 0.22589127,'2010': 0.28829848,'2011': 0.41989028}
    beam_errs = {}
    for name in names:
        for year in years:
            beam_err_file ='spt_Beam_errors/errgrid_'+name+'_'+year+'_'+'150'+'.txt' 
            beam_errs[name+year]=loadtxt(beam_err_file)[:,1]
                                 
    amp_names = names[:2]+[names[2:][i]+years[j] for i in range(5) for j in range(4)]      		
    
    for year in years:
            for k in amp_names[:2]:
                if year == '2008':
                    beam_errs[k] = beam_errs[k+year]*year_weights[year]                    
                else:
                    beam_errs[k] += beam_errs[k+year]*year_weights[year]
        
    correlated = zeros([lmax,lmax])
            
    uncorrelated_root_weight = zeros([lmax,lmax])
            
    uncorrelated = zeros([lmax,lmax])       
            
            
    if freqs[1] == 150:
            for k in amp_names[:2]:         
                correlated += outer((1+beam_errs[k][:lmax])**(-2)-1,(1+beam_errs[k][:lmax])**(-2)-1)
            for k in amp_names[2:14]:
                uncorrelated_root_weight += outer(((1+ beam_errs[k][:lmax])**(-2)-1)*year_weights[k[-4:]]**.5,
                                                      ((1+ beam_errs[k][:lmax])**(-2)-1)*year_weights[k[-4:]]**.5)
            for k in amp_names[14:]:
                uncorrelated += outer(((1+beam_errs[k][:lmax])**(-2)-1)*year_weights[k[-4:]],
                                         ((1+beam_errs[k][:lmax])**(-2)-1)*year_weights[k[-4:]])
                        
    else:
            for k in amp_names[:2]:         
                correlated += outer((1+beam_errs[k][:lmax])**(-1)-1,(1+beam_errs[k][:lmax])**(-2)-1)
            for k in amp_names[2:14]:
                uncorrelated_root_weight += outer(((1+ beam_errs[k][:lmax])**(-1)-1)*year_weights[k[-4:]]**.5,
                                                      ((1+ beam_errs[k][:lmax])**(-1)-1)*year_weights[k[-4:]]**.5)
            for k in amp_names[14:]:
                uncorrelated += outer(((1+beam_errs[k][:lmax])**(-1)-1)*year_weights[k[-4:]],
                                         ((1+beam_errs[k][:lmax])**(-1)-1)*year_weights[k[-4:]])
    
    return correlated+uncorrelated_root_weight+uncorrelated
        
class fgs_model():
    def __init__(self,freqs): 
            
        self.sz_template = hstack([[0],loadtxt("spt_lowl_templates/SZ_template.txt")[:,1]])
        self.poisson_template = hstack([[0],loadtxt("spt_lowl_templates/poisson_template.txt")[:,1]])
        self.cluster_template = hstack([[0,0],loadtxt("spt_lowl_templates/cluster_template.txt")[:,1]])
    
        
        if freqs[1] == 150:
                self.tsz_dep = 1
                self.cib_dep = 1
                
        else:
                self.tsz_dep = self.tSZ_freq_dep(*freqs)
                self.cib_dep = self.cib_freq_dep(*freqs)
    
    def G(self,nu):
        h=6.63*10**-34
        Tcmb=2.725
        kb=1.38*10**-23
        x=h*nu*10**9/kb/Tcmb
        return ((nu*10**9)**3)*x*exp(x)/((exp(x)-1)**2)
    def B(self,nu):
        h=6.63*10**-34
        kb=1.38*10**-23
        Td=20
        x=h*nu*10**9/kb/Td
        return (nu*10**9)**3/(exp(x)-1)
    def cib_freq_dep(self,fr1,fr2,fr0=150):
        b=2.0
        t1,t2,t0 = map(lambda fr: (lambda x0: ((x0*10**9)**b)*self.B(x0)/self.G(x0))(fr),[fr1,fr2,fr0])
        return t1*t2/t0**2
    def tSZ_freq_dep(self,fr1,fr2,fr0=150):
        t1,t2,t0 = map(lambda fr: (lambda x0: x0*(exp(x0)+1)/(exp(x0)-1) - 4)(fr/56.78),[fr1,fr2,fr0])
        return t1*t2/t0**2
    
    def __call__(self,fit_params,lmax):
            
            
            return ((fit_params['Asz']-2.9)*self.tsz_dep + 2.9) * self.sz_template[:lmax] + \
                   fit_params['Aps'] * self.poisson_template[:lmax] * self.cib_dep + \
                   fit_params['Acib'] * self.cluster_template[:lmax] * self.cib_dep
                
    


def CMB_param_estimator(like, start, method = 'powell', options = None):
    
    start = dot(inv(like.cho_param_cov),start)
    
    best_fit_params=minimize(like, start, method = method, options=options)
    
    best_fit_params.x = dot(like.cho_param_cov,best_fit_params.x)
                                    
    output = dict(zip(like.param_names,list(best_fit_params.x)))
    return output

if spectra == '150x150':
        start = [  1.04037203,   0.02264724,   0.13110934,   1.93666773,
         0.9254644 ,   0.06416065,   5.72741826,  20.0334358 ,   5.3754613 ]

elif spectra =='150x143':
        start = [  1.03958287,   0.02247682,   0.13569471,   1.93508836,
         0.93324658,   0.06348503,   6.32184949,  19.77749923,   5.58548061]
else:
        start = [  1.03924319,   0.0221666 ,   0.14439015,   1.94318957,
         0.94628772,   0.06790312,   6.63191637,  20.09969542,   5.25854533]



results = None
j=0
for i in arange(sim_start,sim_end+1):
    j+=1
        best=CMB_param_estimator(like(spectra,str(i),lmax = lmax),
                         start,
                         method = 'Nelder-Mead',
                         options = {'disp': True, 'xtol':1e-4, 'ftol':1e-4,
                                   })
    
        print best
        print "percent complete"
        print j/((sim_end-sim_start)*100.0), '%'
        if results == None: 
            results = best
            for k,v in results.iteritems():
                results[k] = [v]
        else:    
            for k,v in best.iteritems():
                results[k].append(v)
if i = 399:    
    json.dump(results, open("params_output/sim_params"+spectra+"sims"+str(sim_start)+"_"+str(sim_end)+"_lmax_"+str(lmax)+".txt",'w'))
    
    data_best=CMB_param_estimator(like(spectra,str(i),lmax=lmax, use_data=True),
                         start,
                         method = 'Nelder-Mead',
                         options = {'disp': True, 'xtol':1e-4, 'ftol':1e-4,
                                   })
    
    print data_best
    results = data_best
    for k,v in results.iteritems():
        results[k] = [v]
    json.dump(results, open("params_output/data_params"+spectra+str(lmax)+".txt",'w'))
print "Done with this batch"  
