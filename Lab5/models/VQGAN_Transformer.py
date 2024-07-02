import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens'] # 256
        self.mask_token_id = configs['num_codebook_vectors'] # 1024
        self.choice_temperature = configs['choice_temperature']# 4.5
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path), strict=True)
        
    def create_clustered_mask_random_walk(self, shape, cluster_size, num_clusters):
        """
        Create a clustered mask using random walks.

        Parameters:
        shape (tuple): The shape of the mask (height, width).
        cluster_size (int): The desired size of each cluster.
        num_clusters (int): The number of clusters.

        Returns:
        torch.Tensor: A boolean mask with clustered True values.
        """
        def inside(i, j):
            return 0 <= i < height and 0 <= j < width

        mask = torch.zeros(shape, dtype=torch.bool, device='cpu')
        height, width = shape

        for _ in range(num_clusters):
            # Randomly choose a starting point
            start_y = torch.randint(0, height, (1,)).item()
            start_x = torch.randint(0, width, (1,)).item()
            while(mask[start_y, start_x] == True):
                start_y = torch.randint(0, height, (1,)).item()
                start_x = torch.randint(0, width, (1,)).item()
            # Perform a random walk
            y, x = start_y, start_x
            for _ in range(cluster_size//num_clusters):
                mask[y, x] = True
                direction = torch.randint(0, 4, (1,)).item()
                if direction == 0 and y > 0:        # Move up
                    y -= 1
                elif direction == 1 and y < height - 1: # Move down
                    y += 1
                elif direction == 2 and x > 0:      # Move left
                    x -= 1
                elif direction == 3 and x < width - 1: # Move right
                    x += 1

        return mask

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        else:
            raise NotImplementedError


##TODO2 step1-3:            
    def forward(self, x, cur_epoch, total_epoch, cluster_mask=False):
        _, z_indices = self.encode_to_z(x)
        if cluster_mask:
            mask = self.create_clustered_mask_random_walk((16, 16), 100, int((1 - (cur_epoch / total_epoch)) * 8)).flatten()
            mask.view(z_indices.shape).to(z_indices.device)
        else:
            r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
            sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
            mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
            mask.scatter_(dim=1, index=sample, value=True)
        
        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
    
        a_indices = mask * z_indices + (~mask) * masked_indices
        
        
        logits = self.transformer(a_indices)

        return logits, z_indices
    
            

    @torch.no_grad()
    def inpainting(self, z_indices, mask_b, ratio, true_scheduling=False, mask_original=None):
        z_indices = torch.where(mask_b, self.mask_token_id, z_indices)
        logits = self.transformer(z_indices) # [1, 256, 1025]
        #Apply softmax to convert logits into a probability distribution across the last dimension.
        probs = torch.softmax(logits, dim=-1)  # [1, 256, 1025]
        infinite_mask = torch.full_like(probs, float('inf'))
        probs = torch.where(mask_b.unsqueeze(-1), probs, infinite_mask)
        #FIND MAX probability for each token value
        z_indices_predict_prob, z_indices_predict = probs.max(dim=-1) # [1, 256]
        # '''Out of bound handling'''
        # top2_probs, top2_indices = probs.topk(2, dim=-1)  # [1, 256, 2], [1, 256, 2]
    
        # # Get the maximum probability and corresponding index
        # z_indices_predict_prob, z_indices_predict = top2_probs[:, :, 0], top2_indices[:, :, 0]  # [1, 256], [1, 256]
        
        # # Replace out of bound indices (1024) with the second largest index
        # out_of_bound_mask = (z_indices_predict == 1024)
        # z_indices_predict = torch.where(out_of_bound_mask, top2_indices[:, :, 1], z_indices_predict)
        # z_indices_predict_prob = torch.where(out_of_bound_mask, top2_probs[:, :, 1], z_indices_predict_prob)
        # ''''''
        ratio = self.gamma(ratio)
        #predicted probabilities add temperature annealing gumbel noise as confidence
        # g = torch.rand_like(z_indices_predict_prob)   # gumbel noise
        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to("cuda")
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * g
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        if true_scheduling:
            mask_len = torch.unsqueeze(torch.floor(mask_original.sum().view([1]) * ratio), 1)
        else:
            mask_len = torch.unsqueeze(torch.floor(mask_b.sum().view([1]) * ratio), 1)
        mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(mask_b, dim=-1, keepdim=True)-1, mask_len))
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off) # preserve the max confidence part
        
        z_indices_predict = torch.where(mask_b, z_indices_predict, z_indices)
        # z_indices_predict = torch.where(masking, z_indices, z_indices_predict)
        z_indices_predict = torch.clamp(z_indices_predict, 0, 1023)
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
        return z_indices_predict, masking
    
    
    def generate_perlin_noise_2d(shape, res):
        def f(t):
            return 6*t**5 - 15*t**4 + 10*t**3
        
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
        angles = 2 * np.pi * np.random.rand(res[0]+1, res[1]+1)
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.stack((grid[:,:,0]-1, grid[:,:,1]), 2) * g10, 2)
        n01 = np.sum(np.stack((grid[:,:,0], grid[:,:,1]-1), 2) * g01, 2)
        n11 = np.sum(np.stack((grid[:,:,0]-1, grid[:,:,1]-1), 2) * g11, 2)
        t = f(grid)
        return np.sqrt(2)*((1-t[:,:,0])*(1-t[:,:,1])*n00 + t[:,:,0]*(1-t[:,:,1])*n10 + (1-t[:,:,0])*t[:,:,1]*n01 + t[:,:,0]*t[:,:,1]*n11)

    # Function to create a clustered mask using Perlin noise with adjustable resolution
    def create_clustered_mask_perlin(shape, res, cluster_size, threshold=0.5):
        noise = generate_perlin_noise_2d(shape, res)
        initial_mask = noise > threshold
        
        # Flatten the mask to easily adjust the cluster size
        flat_mask = initial_mask.flatten()
        num_true = flat_mask.sum()

        # If the initial mask has more True values than required, randomly turn some to False
        if num_true > cluster_size:
            true_indices = np.where(flat_mask)[0]
            np.random.shuffle(true_indices)
            flat_mask[true_indices[cluster_size:]] = False
        # If the initial mask has fewer True values than required, randomly turn some False values to True
        elif num_true < cluster_size:
            false_indices = np.where(~flat_mask)[0]
            np.random.shuffle(false_indices)
            flat_mask[false_indices[:cluster_size - num_true]] = True
        
        # Reshape the flat mask back to the original shape
        final_mask = flat_mask.reshape(shape)
        return torch.tensor(final_mask, dtype=torch.bool)

    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
# import torch 
# import torch.nn as nn
# import yaml
# import os
# import math
# import numpy as np
# from .VQGAN import VQGAN
# from .Transformer import BidirectionalTransformer


# #TODO2 step1: design the MaskGIT model
# class MaskGit(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
#         self.num_image_tokens = configs['num_image_tokens'] # 256
#         self.mask_token_id = configs['num_codebook_vectors'] # 1024
#         self.choice_temperature = configs['choice_temperature']# 4.5
#         self.gamma = self.gamma_func(configs['gamma_type'])
#         self.transformer = BidirectionalTransformer(configs['Transformer_param'])

#     def load_transformer_checkpoint(self, load_ckpt_path):
#         self.transformer.load_state_dict(torch.load(load_ckpt_path), strict=False)

#     @staticmethod
#     def load_vqgan(configs):
#         cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
#         model = VQGAN(cfg['model_param'])
#         model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
#         model = model.eval()
#         return model
    
# ##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
#     @torch.no_grad()
#     def encode_to_z(self, x):
#         quant_z, indices, _ = self.vqgan.encode(x)
#         indices = indices.view(quant_z.shape[0], -1)
#         return quant_z, indices
    
# ##TODO2 step1-2:    
#     def gamma_func(self, mode="cosine"):
#         """Generates a mask rate by scheduling mask functions R.

#         Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
#         During training, the input ratio is uniformly sampled; 
#         during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
#         Based on experiements, we find that masking more in training helps.
        
#         ratio:   The uniformly sampled ratio [0, 1) as input.
#         Returns: The mask rate (float).

#         """
#         if mode == "linear":
#             return lambda r: 1 - r
#         elif mode == "cosine":
#             return lambda r: np.cos(r * np.pi / 2)
#         elif mode == "square":
#             return lambda r: 1 - r ** 2
#         else:
#             raise NotImplementedError


# ##TODO2 step1-3:            
#     def forward(self, x):
#         _, z_indices = self.encode_to_z(x)
#         r = math.floor(self.gamma_func(np.random.uniform()) * z_indices.shape[1])
#         sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
#         mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
#         mask.scatter_(dim=1, index=sample, value=True)
        
#         masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
    
#         a_indices = mask * z_indices + (~mask) * masked_indices
        
        
#         logits = self.transformer(a_indices)

#         return logits, z_indices
    
# ##TODO3 step1-1: define one iteration decoding   

#     @torch.no_grad()
#     def sample_good(self, masked_indices, num=1, T=11, mode="cosine", temperature=1.0):
#         N = self.num_image_tokens
#         num_masked_tokens = torch.sum(masked_indices == self.mask_token_id, dim=-1)
#         gamma = self.gamma_func(mode)
#         inputs = masked_indices
#         t = 0
#         while (inputs == self.mask_token_id).any():
#             logits = self.transformer(inputs)
#             sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
#             unknown_map = (inputs == self.mask_token_id)
            
#             ratio = (t + 1) / T
#             ratio = gamma(ratio)
            
#             probs = torch.softmax(logits, dim=-1)
#             selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)
#             selected_probs = torch.where(unknown_map, selected_probs, torch.Tensor([torch.inf]).to("cuda"))
            
#             mask_len = torch.unsqueeze(torch.floor(num_masked_tokens * ratio), 1)
#             mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))
            
#             confidence = torch.log(selected_probs) + temperature * torch.rand_like(selected_probs)
#             sorted_confidence, _ = torch.sort(confidence, dim=-1)
#             cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
#             masking = (confidence < cut_off)
#             inputs = torch.where(masking, self.mask_token_id, sampled_ids)
#             t += 1
#         return inputs
            

#     @torch.no_grad()
#     def inpainting(self, image, mask_b, ratio):
#         _, masked_indices = self.encode_to_z(image)
#         logits = self.transformer(masked_indices) # [1, 256, 1025]
#         #Apply softmax to convert logits into a probability distribution across the last dimension.
#         probs = torch.softmax(logits, dim=-1)  # [1, 256, 1025]
#         infinite_mask = torch.full_like(probs, -float('inf'))
#         probs = torch.where(mask_b.unsqueeze(-1), probs, infinite_mask)
#         #FIND MAX probability for each token value
#         z_indices_predict_prob, z_indices_predict = probs.max(dim=-1) # [1, 256]

#         ratio = self.gamma(ratio)
#         #predicted probabilities add temperature annealing gumbel noise as confidence
#         g = torch.rand_like(z_indices_predict_prob)   # gumbel noise
#         temperature = self.choice_temperature * (1 - ratio)
#         confidence = z_indices_predict_prob + temperature * g
        
#         z_indices_predict = self.sample_good(masked_indices)
#         #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
#         #sort the confidence for the rank 
#         #define how much the iteration remain predicted tokens by mask scheduling
#         #At the end of the decoding process, add back the original token values that were not masked to the predicted tokens
#         mask_bc=mask_b
#         return z_indices_predict, mask_bc
    
# __MODEL_TYPE__ = {
#     "MaskGit": MaskGit
# }
    


        
