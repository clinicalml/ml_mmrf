import torch
import torch.nn as nn
import numpy as np
import logging
from lifelines.utils import concordance_index
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset
from torchcontrib.optim import SWA

class Model(nn.Module):
    def __init__(self):
        torch.manual_seed(0)
        np.random.seed(0)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        super(Model, self).__init__()
        
    def predict(self,**kwargs):
        raise ValueError('Should be overriden')
    def forward(self,**kwargs):
        raise ValueError('Should be overriden')
    
    def calc_stats(self, preds, data_loader):
        B, X, A, M, Y, CE = data_loader.dataset.tensors
        if Y.shape[-1]>1:
            Y_oh      = Y.detach().cpu().numpy()
            bin_preds = self.prediction.detach().cpu().numpy()
            Y_np      = bin_preds[np.argmax(Y_oh,-1)]
        else:
            Y_np      = Y.detach().cpu().numpy().ravel()
        CE_np     = CE.detach().cpu().numpy().ravel()
        preds_np  = preds.detach().cpu().numpy().ravel()
        event_obs = (1.-CE_np).ravel()
        idx       = np.where(event_obs>0)[0]
        mse  = np.square(Y_np[idx]-preds_np[idx]).mean()
        r2   = r2_score(Y_np[idx], preds_np[idx])
        ci   = concordance_index(Y_np, preds_np, event_obs)
        return mse, r2, ci
    
    def calc_stats_unsupervised(self, preds, data_loader):
        B, X, A, M, Y, CE = data_loader.dataset.tensors
        CE_np     = CE.detach().cpu().numpy().ravel()
        X_np      = X.detach().cpu().numpy()
        M_np      = M.detach().cpu().numpy()
        preds_np  = preds.detach().cpu().numpy()
        event_obs = (1.-CE_np).ravel()
        idx       = np.where(event_obs>0)[0]
        diff_sq   = np.square(X_np[idx][:,1:,:]-preds_np[idx]).sum(-1)
        mse       = (diff_sq*M_np[idx,1:]).mean()
        return mse
    
    def collect_best_params(self, best_params):
        for n,p in self.named_parameters():
            best_params[n] = p.clone().detach()
    
    def set_params(self, best_params):
        for n, p in self.named_parameters():
            p.data.copy_(best_params[n])
    
    def get_masks(self, M):
        m_t    = ((torch.flip(torch.cumsum(torch.flip(M.sum(-1), (1,)), 1), (1,))>1.)*1)
        m_g_t  = (m_t.sum(-1)>1)*1.
        lens   = m_t.sum(-1)
        return m_t, m_g_t, lens
    
    def masked_gaussian_nll_3d(self, x, mu, std, mask):
        nll        = 0.5*np.log(2*np.pi) + torch.log(std)+((mu-x)**2)/(2*std**2)
        masked_nll = (nll*mask)
        return masked_nll
    
    def apply_reg(self, p, reg_type='l2'):
        if reg_type == 'l1':
            return torch.sum(torch.abs(p))
        elif reg_type=='l2':
            return torch.sum(p.pow(2))
        else:
            raise ValueError('bad reg')
            
    def fit_unsupervised(self, train_loader, valid_loader, epochs, lr, eval_freq=20, print_freq=1000, anneal = None, fname = None, opt_type = 'ADAM', imp_sampling=False, semi_amortized_steps = 0, nsamples = 1):
        if opt_type == 'ADAM': 
            opt     = torch.optim.Adam(self.parameters(), lr=lr)
            if semi_amortized_steps>0:
                try:
                    opt     = torch.optim.Adam(self.parameters(), lr=lr)
                    inf_opt = torch.optim.Adam(self.inf_network.parameters(), lr=lr)
                except: 
                    raise ValueError('Cannot optimize variational parameters for the selected model')
                    
        elif opt_type == 'SWA': 
            base_opt = torch.optim.Adam(self.parameters(), lr=lr)
            opt = SWA(base_opt, swa_start=100, swa_freq=50, swa_lr=lr)
            if semi_amortized_steps>0:
                raise ValueError('not implemented for SWA')
        else: 
            raise ValueError('bad opt type...')
            
        best_nelbo, best_nll, best_kl, best_ep = 100000, 100000, 100000, -1
        best_params = {}
        if fname is not None: 
            logging.basicConfig(
                filename=fname[:-4]+'_loss.log', filemode='w',
                format='%(asctime)s - %(levelname)s \t %(message)s',
                level=logging.INFO)
        if anneal is None: 
            anneal = epochs/10
        for epoch in range(1, epochs+1):
            anneal = min(1, epoch/(epochs*0.5))
            self.train()
            batch_loss = 0
            idx        = 0
            for data_tuples in train_loader:
                # Updates to inference network only
                if semi_amortized_steps>0:
                    losslist = []
                    for k in range(semi_amortized_steps):
                        inf_opt.zero_grad()
                        _, loss  = self.forward_unsupervised(*data_tuples, anneal = 1.)
                        loss.backward()
                        inf_opt.step()
                        losslist.append(loss.item())
                    if epoch%print_freq==0:
                        print ('Tightening variational params: ',np.array(losslist))
                    anneal = 1.
                opt.zero_grad()
                if nsamples>1:
                    dt = [k.repeat(nsamples,1) if k.dim()==2 else k.repeat(nsamples,1,1) for k in data_tuples]
                else:
                    dt = data_tuples
                _, loss  = self.forward_unsupervised(*dt, anneal = anneal)
                loss.backward()
                """
                for n,p in self.named_parameters():
                    if np.any(np.isnan(p.grad.cpu().numpy())) or np.any(np.isinf(p.grad.cpu().numpy())):
                        print (n,'is nan or inf')
                        import ipdb;ipdb.set_trace()
                        print ('stop')
                """
                opt.step()
                idx +=1
                batch_loss += loss.item()
            if epoch%eval_freq==0:
                self.eval()
                (nelbo, nll, kl,_), _ = self.forward_unsupervised(*valid_loader.dataset.tensors, anneal = 1.)
                if imp_sampling: 
                    batch_nll      = []
                    for i, valid_batch_loader in enumerate(valid_loader): 
                        nll_estimate   = self.imp_sampling(*valid_batch_loader, nelbo, anneal = 1.)
                        nll_estimate   = nll_estimate.item()
                        batch_nll.append(nll_estimate)
                    nll_estimate = np.mean(batch_nll)
                nelbo, nll, kl = nelbo.item(), nll.item(), kl.item()
                if nelbo<best_nelbo:
                    best_nelbo  = nelbo; best_nll = nll; best_kl = kl; best_ep = epoch
                    if imp_sampling: 
                        best_nll_estimate = nll_estimate
                    self.collect_best_params(best_params)
                    if fname is not None:
                        if opt_type == 'SWA': 
                            opt.swap_swa_sgd()
                        torch.save(self.state_dict(), fname)
                self.train()
            if epoch%print_freq==0:
                if imp_sampling: 
                    print ('Ep',epoch,' Loss:',batch_loss/float(idx),', Anneal:', anneal, \
                    ', Best NELBO:%.3f, NLL est.:%.3f, NLL:%.3f, KL: %.3f @ epoch %d'%(best_nelbo, \
                        best_nll_estimate, best_nll, best_kl, best_ep))
                else: 
                    print ('Ep',epoch,' Loss:',batch_loss/float(idx),', Anneal:', anneal, \
                        ', Best NELBO:%.3f, NLL:%.3f, KL: %.3f @ epoch %d'%(best_nelbo, \
                            best_nll, best_kl, best_ep))
                # print ('Ep',epoch,' Loss:',batch_loss/float(idx),', Anneal:', anneal, ', Best NELBO:%.3f NLL:%.3f, KL: %.3f @ epoch %d'%(best_nelbo, best_nll, best_kl, best_ep))
                if fname is not None: 
                    msg = 'Ep: %d, Loss: %f, Anneal: %.3f, Best NELBO:%.3f NLL:%.3f, KL: %.3f @ epoch %d'
                    logging.info(msg, epoch, batch_loss/float(idx), anneal, best_nelbo, best_nll, best_kl, best_ep)
        print ('Best NELBO:%.3f, NLL:%.3f, KL:%.3f@ epoch %d'%(best_nelbo, best_nll, best_kl, best_ep))
        self.best_params   = best_params
        self.best_nelbo    = best_nelbo
        self.best_nll      = best_nll
        self.best_kl       = best_kl
        self.best_ep       = best_ep
        return best_params, best_nelbo, best_nll, best_kl, best_ep
    

    def fit(self, train_loader, valid_loader, epochs, lr, loss_type = 'sup', eval_freq=20, print_freq=1000, 
            track_val = 'ci', anneal=None, fname = None, only_supervised_params = False):
        if only_supervised_params:
            print ('restricting to updating predictive parameters only')
            opt = torch.optim.Adam([p for n,p in self.named_parameters() if 'unsup_model' not in n], lr=lr)
        else:
            opt = torch.optim.Adam(self.parameters(), lr=lr)
        best_mse, best_r2, best_ci, best_ep = 100, -10, 0., -1.
        best_nelbo, best_nll, best_kl = 100000, 100000, 100000
        mse, r2, ci = 100,-1,-1
        best_params = {}
        if fname is not None: 
            logging.basicConfig(
                filename=fname[:-4]+'_loss.log', filemode='w',
                format='%(asctime)s - %(levelname)s \t %(message)s',
                level=logging.INFO)
        if anneal is None: 
            anneal = epochs/10
        for epoch in range(1, epochs+1):
            self.train()
            batch_loss = 0
            idx = 0
            for data_tuples in train_loader:
                opt.zero_grad()
                if loss_type == 'sup':
                    _, loss  = self.predict(*data_tuples)
                elif loss_type == 'both':
                    _, loss  = self.forward(*data_tuples, anneal = anneal)
                loss.backward()
                opt.step()
                idx +=1
                batch_loss += loss.item()
            if epoch%eval_freq==0:
                self.eval()
                preds, ploss_ = self.predict(*valid_loader.dataset.tensors)
                mse, r2, ci = self.calc_stats(preds, valid_loader)
                
                if loss_type == 'sup': 
                    loss_ = ploss_.item() 
                elif loss_type == 'both': 
                    _, loss_ = self.forward(*valid_loader.dataset.tensors, anneal = 1.)
                    loss_ = loss_.item()
                    
                if loss_ < best_nelbo: 
                    best_nelbo = loss_
                if track_val == 'mse' and mse<best_mse:
                    best_ci  = ci; best_mse = mse; best_r2 = r2; best_ep = epoch
                    self.collect_best_params(best_params)
                    if fname is not None:
                        torch.save(self.state_dict(), fname)
                elif track_val =='ci' and ci>best_ci:
                    best_ci  = ci; best_mse = mse; best_r2 = r2; best_ep = epoch
                    self.collect_best_params(best_params)
                    if fname is not None:
                        torch.save(self.state_dict(), fname)
                self.train()
            if epoch%print_freq==0:
                print ('Ep', epoch, 'Tr-Loss: %.3f, Va-Loss: %.3f, Best Loss: %.3f, Best MSE:%.3f R2:%.3f, CI: %.3f @ epoch %d'% \
                       (batch_loss/float(idx), loss_,  best_nelbo, best_mse, best_r2, best_ci, best_ep))
                if fname is not None: 
                    msg = 'Ep: %d, Tr-Loss: %.3f, Va-Loss: %.3f, Best Loss (held out): %.3f, Best MSE:%.3f, R2:%.3f, CI: %.3f @ epoch %d'
                    logging.info(msg, epoch, batch_loss/float(idx), loss_, best_nelbo, best_mse, best_r2, best_ci, best_ep)
        print ('Best MSE:%.3f R2:%.3f, CI: %.3f @ epoch %d'%(best_mse, best_r2, best_ci, best_ep))
        self.best_params = best_params
        self.best_mse    = best_mse
        self.best_r2     = best_r2
        self.best_ci     = best_ci
        self.best_ep     = best_ep
        return best_params, best_mse, best_r2, best_ci, best_ep

def setup_torch_dataset(dataset, fold, tvt, device, batch_size = 600, forward_fill = False):
    B  = torch.from_numpy(dataset[fold][tvt]['b'].astype('float32')).to(device)
    X  = torch.from_numpy(dataset[fold][tvt]['x'].astype('float32')).to(device)
    A  = torch.from_numpy(dataset[fold][tvt]['a'].astype('float32')).to(device)
    M  = torch.from_numpy(dataset[fold][tvt]['m'].astype('float32')).to(device)
    
    y_vals   = dataset[fold][tvt]['ys_seq'][:,0].astype('float32')
    idx_sort = np.argsort(y_vals)
    if 'digitized_y' in dataset[fold][tvt]:
        print ('using digitized y')
        Y  = torch.from_numpy(dataset[fold][tvt]['digitized_y'].astype('float32')).to(device)
    else:
        Y  = torch.from_numpy(dataset[fold][tvt]['ys_seq'][:,[0]].astype('float32')).to(device)
    CE = torch.from_numpy(dataset[fold][tvt]['ce'].astype('float32')).to(device)
    
    data        = TensorDataset(B[idx_sort], X[idx_sort], A[idx_sort], M[idx_sort], Y[idx_sort], CE[idx_sort])
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return data, data_loader

def pt_numpy(tensor):
    return tensor.detach().cpu().numpy()

if __name__=='__main__':
    pass