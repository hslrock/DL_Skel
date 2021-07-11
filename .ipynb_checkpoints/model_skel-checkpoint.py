import torch.nn as nn

class model_skel(nn.Module):
    
    def __init__(self,ncz,vec,outz):
        '''
            Initialize model's hypermarmeter/structure using argumnets

            ncz:input size

            vec_size: latent size
            outz:output size
        '''
        super(model_skel, self).__init__()
    
    
        #Layers
        self.fc1=nn.Linear(ncz,vec)
        self.fc2=nn.Linear(vec,outz)
        self.tanh=nn.Tanh()
        self.relu=nn.LeakyReLU()
        
        
    def forward(self,x):
        
        #Forward Function
        
        
        x=self.relu(self.fc1(x))
        x=self.tanh(self.fc2(x))
        return x