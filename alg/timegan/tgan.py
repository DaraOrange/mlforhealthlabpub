import numpy as np
import torch
from model import Embedder, Generator, Discriminator, Recovery, Supervisor


def MinMaxScaler(dataX):

    min_val = np.min(np.min(dataX, axis = 0), axis = 0)
    dataX = dataX - min_val

    max_val = np.max(np.max(dataX, axis = 0), axis = 0)
    dataX = dataX / (max_val + 1e-7)

    return dataX, min_val, max_val


#%% Start TGAN function (Input: Original data, Output: Synthetic Data)
def train(dataX, parameters):
	# tf.reset_default_graph()  # https://discuss.pytorch.org/t/how-to-free-graph-manually/9255

	# Basic Parameters
    No = len(dataX)
    data_dim = len(dataX[0][0,:])

    # Maximum seq length and each seq length
    dataT = list()
    Max_Seq_Len = 0
    for i in range(No):
        Max_Seq_Len = max(Max_Seq_Len, len(dataX[i][:,0]))
        dataT.append(len(dataX[i][:,0]))

    # Normalization
    if ((np.max(dataX) > 1) | (np.min(dataX) < 0)):
        dataX, min_val, max_val = MinMaxScaler(dataX)
        Normalization_Flag = 1
    else:
        Normalization_Flag = 0

    # Network Parameters
    hidden_dim   = parameters['hidden_dim']
    num_layers   = parameters['num_layers']
    iterations   = parameters['iterations']
    batch_size   = parameters['batch_size']
    module_name  = parameters['module_name']    # 'lstm' or 'lstmLN'
    z_dim        = parameters['z_dim']
    gamma        = 1
    lr 			 = parameters['lr']

    def random_generator(batch_size, z_dim, T_mb, Max_Seq_Len):
        Z_mb = list()
        for i in range(batch_size):
            Temp = np.zeros([Max_Seq_Len, z_dim])
            Temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
            Temp[:T_mb[i],:] = Temp_Z
            Z_mb.append(Temp_Z)
        return Z_mb

    embedder = Embedder(data_dim, hidden_dim, num_layers).cuda()
    discriminator = Discriminator(hidden_dim, num_layers).cuda()
    generator = Generator(z_dim, hidden_dim, num_layers).cuda()
    recovery = Recovery(hidden_dim, data_dim, num_layers).cuda()
    supervisor = Supervisor(data_dim, num_layers)

    e_opt = torch.optim.Adam(embedder.parameters(), lr)
    r_opt = torch.optim.Adam(recovery.parameters(), lr)
    s_opt = torch.optim.Adam(supervisor.parameters(), lr)
    g_opt = torch.optim.Adam(generator.parameters(), lr)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr)
    
    #%% Embedding Learning
    
    print('Start Embedding Network Training')
    
    for itt in range(iterations):
        e_opt.zero_grad()
        r_opt.zero_grad()
        s_opt.zero_grad()
        g_opt.zero_grad()
        d_opt.zero_grad()
        
        # Batch setting
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X = list(dataX[i] for i in train_idx)
        T = list(dataT[i] for i in train_idx)

        H = embedder(X, T)
        X_tilde = recovery(H, T)
        
        # Generator
        H_hat_supervise = supervisor(H, T)
            
        # Train embedder    
        G_loss_S = torch.nn.MSELoss()(H[:,1:,:], H_hat_supervise[:,1:,:])
        E_loss_T0 = torch.nn.MSELoss()(X, X_tilde)
        E_loss0 = 10*torch.sqrt(E_loss_T0)
        E_loss = E_loss0  + 0.1*G_loss_S

        E_loss0.backward()

        # Update model parameters
        e_opt.step()
        r_opt.step()
        
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', e_loss: ' + str(np.round(np.sqrt(E_loss_T0),4)) )        
            
    print('Finish Embedding Network Training')
    
    print('Start Training with Supervised Loss Only')
    
    for itt in range(iterations):
        e_opt.zero_grad()
        r_opt.zero_grad()
        s_opt.zero_grad()
        g_opt.zero_grad()
        d_opt.zero_grad()
        
        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X = list(dataX[i] for i in train_idx)
        T = list(dataT[i] for i in train_idx)        
        
        H = embedder(X, T)     
        G_loss_S = torch.nn.MSELoss()(H[:,1:,:], H_hat_supervise[:,1:,:])
        G_loss_S.backward()
        s_opt.step()
                           
        if itt % 1000 == 0:
            print('step: '+ str(itt) + ', s_loss: ' + str(np.round(np.sqrt(G_loss_S),4)) )
                
    print('Finish Training with Supervised Loss Only')
    
    print('Start Joint Training')
    
    # Training step
    for itt in range(iterations):

        idx = np.random.permutation(No)
        train_idx = idx[:batch_size]     
            
        X = list(dataX[i] for i in train_idx)
        T = list(dataT[i] for i in train_idx)   

        # Generator Training
        for _ in range(2):
            e_opt.zero_grad()
            r_opt.zero_grad()
            s_opt.zero_grad()
            g_opt.zero_grad()
            d_opt.zero_grad()

            H = embedder(X, T)
            H_hat_supervise = supervisor(H, T)
            X_tilde = recovery(H, T)

            E_hat = generator(Z, T)
            H_hat = supervisor(E_hat, T)

            X_hat = recovery(H_hat, T)

            Y_fake = discriminator(H_hat, T)        # Output of supervisor
            Y_fake_e = discriminator(E_hat, T)      # Output of generator

            G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
            G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

            G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

            G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
            G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))

            G_loss_V = G_loss_V1 + G_loss_V2

            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

            G_loss.backward()
            g_opt.step()
            s_opt.step()


            H = embedder(X, T)
            X_tilde = recovery(H, T)

            H_hat_supervise = supervisor(H, T)
            G_loss_S = torch.nn.functional.mse_loss(
                H_hat_supervise[:,:-1,:], 
                H[:,1:,:]
            ) # Teacher forcing next output

            E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, X)
            E_loss0 = 10 * torch.sqrt(E_loss_T0)
            E_loss = E_loss0 + 0.1 * G_loss_S

            E_loss.backward()
            e_opt.step()
            r_opt.step()


        # Random Generator
        Z = torch.rand((batch_size, Max_Seq_Len, z_dim))

        H = embedder(X, T).detach()
        H_hat = supervisor(H, T).detach()
        E_hat = generator(Z, T).detach()

        # Forward Pass
        Y_real = discriminator(H, T)            # Encoded original data
        Y_fake = discriminator(H_hat, T)        # Output of supervisor
        Y_fake_e = discriminator(E_hat, T)      # Output of generator

        D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
        D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
        D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
        
        D_loss.backward()
        d_opt.step()
   
    
    print('Finish Joint Training')
