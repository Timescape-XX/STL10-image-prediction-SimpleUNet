import torch
import torch.nn as nn

device="cuda" if torch.cuda.is_available() else "cpu"

"""
Diffusion schedule parameters
"""
T=1000
betas=torch.linspace(1e-4,0.02,T).to(device) # linear schedule
alphas=(1.0-betas).to(device)
alphas_cumprod=torch.cumprod(alphas,dim=0).to(device)
sqrt_alphas_cumprod=alphas_cumprod.sqrt().to(device)
sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0-alphas_cumprod).to(device)

# forward diffusion q(x_t|x_{t-1}) variance
alphas_cumprod_prev=torch.cat([torch.tensor([1.0],device=device),alphas_cumprod[:-1]],dim=0)
posterior_variance=betas*(1.0-alphas_cumprod_prev)/(1.0-alphas_cumprod).to(device)

def get_index_from_list(vals,t,shape):
    batch_size=t.shape[0]
    out=vals.gather(-1,t)
    return out.reshape(batch_size, *((1,)*(len(shape)-1))).to(t.device)

def forward_diffusion_sample(x_0,t,device):

    # noise addition
    noise=torch.randn_like(x_0).to(device)
    sqrt_alphas_cumprod_t=get_index_from_list(sqrt_alphas_cumprod,t,x_0.shape)
    sqrt_one_minus_alphas_cumprod_t=get_index_from_list(sqrt_one_minus_alphas_cumprod,t,x_0.shape)
    return sqrt_alphas_cumprod_t*x_0.to(device)+sqrt_one_minus_alphas_cumprod_t*noise.to(device),noise

def sample_timesteps(x_t,t,model,label):
    """
    single step for reverse diffusion p(x_{t-1}|x_t)
    """
    betas_t=get_index_from_list(betas,t,x_t.shape)
    sqrt_one_minus_alphas_cumprod_t=get_index_from_list(sqrt_one_minus_alphas_cumprod,t,x_t.shape)
    sqrt_recip_alphas_t=get_index_from_list(torch.sqrt(1.0/alphas),t,x_t.shape)

    # predict noise of current step
    model_mean=sqrt_recip_alphas_t*(x_t-(betas_t/sqrt_one_minus_alphas_cumprod_t)*model(x_t,t,label))

    if t[0]==0:
        return model_mean
    else:
        posterior_variance_t=get_index_from_list(posterior_variance,t,x_t.shape)
        noise=torch.randn_like(x_t).to(device)
        return model_mean+torch.sqrt(posterior_variance_t)*noise

def sample(model,label,image_size=96,num_channels=3,batch_size=1):
    """
    full reverse diffusion process p(x_{t-1}|x_t)
    """
    model.eval()
    with torch.no_grad():
        # noise initialization
        x=torch.randn((batch_size,num_channels,image_size,image_size),device=device)

        # start reverse diffusion
        for i in reversed(range(T)):
            t=torch.full((batch_size,),i,dtype=torch.long,device=device)
            x=sample_timesteps(x,t,model,label)
    
    x=(x+1.0)/2.0
    x=torch.clamp(x,0.0,1.0)
    return x