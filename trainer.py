import copy

import torch
import torch.optim as optim
import torch.nn.functional as F

from loss_function_helpers import noise_normalize_, noise_regularize, kl_divergence

class Trainer():
    def __init__(self, image, avg_face, models, target, cfg):
        self.models = models.to(cfg.DEVICE)
        self.image = image.to(cfg.DEVICE)
        self.avg_face = avg_face.to(cfg.DEVICE)
        self.target = target.to(cfg.DEVICE)
        self.mean_bgr = torch.FloatTensor([91.4953, 103.8827, 131.0912])
        self.mean_bgr = self.mean_bgr.unsqueeze(0)
        self.mean_bgr = self.mean_bgr.unsqueeze(-1)
        self.mean_bgr = self.mean_bgr.unsqueeze(-1)
        self.mean_bgr = self.mean_bgr.to(cfg.DEVICE)
        n = len(image)

        if cfg.USE_E4E:
            self.latent = self.e4e_latent_prediction(self.image)
            self.start_steps = 300
        else:
            self.latent = models.get_latent(n, trainable=False)
            self.start_steps = 0

        self.noises = models.get_noise(n, trainable=False)

        assert self.latent.size(0) == n
        assert all([x.size(0) == n for x in self.noises])

        self.masks = models.get_image_masks(self.image)
        self.original_masks = copy.deepcopy(self.masks)
        self.cfg = cfg
        self.cfg.LOSS.weights.seg = 0.051
    @property
    def device(self):
        return next(self.models.parameters()).device


    def e4e_latent_prediction(self, image):
        from models.e4e import Encoder
        enc = Encoder().to(image.device)
        latent = enc(image)
        return latent

    def convert_attr_input(self, images_generated):
        images_attr_input = images_generated * 255
        # rgb -> bgr
        images_attr_input = images_attr_input[:,[2, 1, 0],:,:]
        images_attr_input = images_attr_input - self.mean_bgr
        return images_attr_input
        
    def train_latent(self):
        n = self.latent.size(0)
        self.latent.requires_grad_(True)

        size_multiplier = torch.tensor([self.cfg.SIZE]).view(1, -1).to(self.device)
        target_attr_portion = size_multiplier * get_component_portion(self.masks['dynamic'])

        optimizer = optim.Adam([self.latent], lr=self.cfg.OPTIMIZER.LR_LATENT)

        losses = []
        dynamic_masking_start = self.cfg.LOSS.start_steps.seg
        dynamic_masking_end = dynamic_masking_start + self.cfg.DYNAMIC_MASKING_ITERS
        
        #mask_I_in = self.models.face_parser(self.image, mode='mse')

        for i in range(self.start_steps, self.cfg.N_LATENT_STEPS):
            if i % self.cfg.N_ITER_PRINT == 0:
                print(f'Latent optimization {i}/{self.cfg.N_LATENT_STEPS}')

            if self.cfg.OPTIMIZER.REINIT and i == self.cfg.LOSS.start_steps.classf:
                optimizer = optim.Adam([self.latent], lr=self.cfg.OPTIMIZER.LR_LATENT)

            if i == dynamic_masking_end and self.cfg.N_NOISE_STEPS > 0:
                self.noises = self.models.get_noise(n, trainable=False, random=True)

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True) 
            
            loss = {}
            
            loss['mse'] = batch_mse_loss(self.image, images_generated, self.masks['mse'])
            
            #loss['mse'] += batch_mse_loss(self.image, images_generated, self.masks['shape'])
           
            
           
            if self.cfg.LOSS.weights.classf > 0 and i >= self.cfg.LOSS.start_steps.classf:
                #attr_pred = torch.sigmoid(self.models.classifier(images_generated)) # tensor([[0.9380, 0.8953, 0.7418]], device='cuda:0', grad_fn=<SigmoidBackward>)
                G_id = self.models.face_net(images_generated)  # 这里输入不用转换，不能用images_attr_input等经过self.convert_attr_input的作为输�?                
                
                O_id = self.models.face_net(self.image)
                #print("Human_information:",human_information.size())
                
                images_attr_input = self.convert_attr_input(images_generated)
                images_orig_input = self.convert_attr_input(self.image)
                
                attr_pred = self.models.classifier(images_attr_input)
                
                attr_orig_pred = self.models.classifier(images_orig_input)
                #print("attr_orig_pred", attr_orig_pred)
                #print("attr_pred",attr_pred)
                #print("Target:",self.target)
                # print(attr_pred)
                #['Male'0, 'Female'1, 'Young'2, 'Middle Aged'3, 'Senior'4,'Asian'5, 'White'6, 'Black'7,'Black Hair'8]
            
                attr_pred_1 = attr_pred[0][0]
                attr_pred_orig_1 = attr_orig_pred[0][0]
                
                if attr_pred_orig_1>= 0.5:               
                   loss['classf'] = 10*kl_divergence(1.0-attr_pred_1, self.target)
                else:
                   loss['classf'] = 10*kl_divergence(attr_pred_1, self.target)
                
                
                # Identitaion information
                
                
                
                #print(kl_divergence(G_id, O_id))
                   
                attr_young_pred = attr_pred[0][2]
                attr_mid_pred = attr_pred[0][3]
                attr_senior_pred = attr_pred[0][4]
                
                attr_young_orig_pred = attr_orig_pred[0][2]
                attr_mid_orig_pred = attr_orig_pred[0][3]
                attr_senior_orig_pred = attr_orig_pred[0][4]
                
                if attr_young_orig_pred >= 0.45:
                   loss['classf'] += kl_divergence(attr_mid_pred, self.target)
                if attr_mid_orig_pred >= 0.45:
                   loss['classf'] += kl_divergence(attr_senior_pred, self.target)
                if attr_senior_orig_pred >= 0.45:
                   loss['classf'] += kl_divergence(attr_young_pred, self.target)
                   
                attr_asian_pred = attr_pred[0][5]
                attr_white_pred = attr_pred[0][6]
                attr_black_pred = attr_pred[0][7]
                
                attr_asian_orig_pred = attr_orig_pred[0][5]
                attr_white_orig_pred = attr_orig_pred[0][6]
                attr_black_orig_pred = attr_orig_pred[0][7]
                
                if attr_asian_orig_pred >= 0.45:
                   loss['classf'] += kl_divergence(attr_white_pred, self.target)
                if attr_white_orig_pred >= 0.45:
                   loss['classf'] += kl_divergence(attr_black_pred, self.target)
                if attr_black_orig_pred >= 0.45:
                   loss['classf'] += kl_divergence(attr_asian_pred, self.target)                
                #attr_pred_2 = attr_pred[0][2]               
                #loss['classf'] += kl_divergence(attr_pred_2, self.target)
                #attr_pred_3 = attr_pred[0][5]               
                #loss['classf'] += kl_divergence(attr_pred_3, self.target)
                #attr_pred_4 = attr_pred[0][7]               
                #loss['classf'] += 5.0*kl_divergence(attr_pred_4, self.target)
              
            if self.cfg.LOSS.weights.seg > 0 and i >= self.cfg.LOSS.start_steps.seg:
                i_dynamic_mask = self.models.face_parser(images_generated, mode='shape')
                
                loss['seg'] = batch_mse_loss(i_dynamic_mask, self.masks['shape'])
                #loss['seg'] += 100000*kl_divergence(G_id, O_id)
                
                
                print("Human_information loss function :",F.mse_loss(G_id, O_id, reduction='none'))
                
                #i_gen_mask = self.models.face_parser(images_generated, mode='shape')
                #i_avg_mask = self.models.face_parser(self.avg_face, mode='shape')
                
                #loss['seg'] += batch_mse_loss(i_gen_mask, i_avg_mask)
                
                #print("22222222222222222222222222222222222222222222222222222")
                #print("loss['seg']", loss['seg'])

            #if self.cfg.SIZE > 0 and i >= self.cfg.LOSS.start_steps.seg:
            #   i_dynamic_mask = self.models.face_parser(images_generated, mode='shape')
            #   i_img_portion = get_component_portion(i_dynamic_mask)
            #   loss['size'] = kl_divergence(i_img_portion, target_attr_portion)

            loss_sum = 0
            for term in loss:
                   weight = self.cfg.LOSS.weights[term]
                   loss_sum += weight * loss[term].sum()

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()


            #if self.cfg.DYNAMIC_MASKING and dynamic_masking_start <= i < dynamic_masking_end:
                #self.masks = self.models.update_image_masks(images_generated, self.original_masks, self.masks)

        print('Finished latent optimization.\n')
        return losses

    def train_noise(self):
        self.noises = [x.requires_grad_(True) for x in self.noises]
        self.latent = self.latent.detach()
        optimizer = optim.Adam(self.noises, lr=self.cfg.OPTIMIZER.LR_NOISE)

        for i in range(self.cfg.N_NOISE_STEPS+1):
            if i % self.cfg.N_ITER_PRINT == 0:
                print(f'Noise optimization {i}/{self.cfg.N_NOISE_STEPS}')

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
            loss = {}
            loss['n_loss'] = noise_regularize(self.noises)
            noise_normalize_(self.noises)

            loss['mse'] = batch_mse_loss(self.image, images_generated, self.masks['mse'])
            
            #loss['mse'] += batch_mse_loss(self.image, images_generated, self.masks['shape'])


            #mask_I_in = self.models.face_parser(self.image, mode='mse')
            #mask_I_gen = self.models.face_parser(images_generated, mode='mse')
            
            
            #loss['mse'] = batch_mse_loss(mask_I_in, mask_I_gen, self.masks['mse'])


            
            loss_sum = 0
            for term in loss:
                loss_sum += self.cfg.LOSS.weights[term] * loss[term]

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Finished noise optimization.\n')


    @torch.no_grad()
    def generate_result(self):
        images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
        blend_mask = self.models.face_parser(images_generated, 'shape')
        result = images_generated
        #print("1111111111111111111111111111111111")
        #print(blend_mask.shape)
        
        #result = (1 - blend_mask) *images_generated +  blend_mask * self.image
        return result

    def generate_image(self):
        return self.models.generator(self.latent, noise=self.noises, to_01=True)

    @torch.no_grad()
    def generate_comparison(self):
        comparisons = []
        results = self.generate_result()
        for original, res in zip(self.image, results):
            comparisons.append(torch.stack((original, res)))
        comparisons = torch.cat(comparisons, dim=0)
        return comparisons

class NewTrainer():
    def __init__(self, image, models, target, cfg):
        self.models = models.to(cfg.DEVICE)
        self.image = image.to(cfg.DEVICE)
        self.target = target.to(cfg.DEVICE)

        n = len(image)

        if cfg.USE_E4E:
            self.latent = self.e4e_latent_prediction(self.image)
            self.start_steps = 300
        else:
            self.latent = models.get_latent(n, trainable=False)
            self.start_steps = 0

        self.noises = models.get_noise(n, trainable=False)

        assert self.latent.size(0) == n
        assert all([x.size(0) == n for x in self.noises])

        self.masks = models.get_image_masks(self.image)
        self.original_masks = copy.deepcopy(self.masks)
        self.cfg = cfg
        self.mean_bgr = torch.FloatTensor([91.4953, 103.8827, 131.0912])
        self.mean_bgr = self.mean_bgr.unsqueeze(0)
        self.mean_bgr = self.mean_bgr.unsqueeze(-1)
        self.mean_bgr = self.mean_bgr.unsqueeze(-1)
        self.mean_bgr = self.mean_bgr.to(cfg.DEVICE)

    @property
    def device(self):
        return next(self.models.parameters()).device


    def e4e_latent_prediction(self, image):
        from models.e4e import Encoder
        enc = Encoder().to(image.device)
        latent = enc(image)
        return latent

    def convert_attr_input(self, images_generated):
        images_attr_input = images_generated * 255
        # rgb -> bgr
        images_attr_input = images_attr_input[:,[2, 1, 0],:,:]
        images_attr_input = images_attr_input - self.mean_bgr
        return images_attr_input

    def train_latent(self):
        n = self.latent.size(0)
        self.latent.requires_grad_(True)

        size_multiplier = torch.tensor([self.cfg.SIZE]).view(1, -1).to(self.device)
        target_attr_portion = size_multiplier * get_component_portion(self.masks['dynamic'])

        optimizer = optim.Adam([self.latent], lr=self.cfg.OPTIMIZER.LR_LATENT)

        losses = []
        dynamic_masking_start = self.cfg.LOSS.start_steps.seg
        dynamic_masking_end = dynamic_masking_start + self.cfg.DYNAMIC_MASKING_ITERS

        for i in range(self.start_steps, self.cfg.N_LATENT_STEPS):
            if i % self.cfg.N_ITER_PRINT == 0:
                print(f'Latent optimization {i}/{self.cfg.N_LATENT_STEPS}')

            if self.cfg.OPTIMIZER.REINIT and i == self.cfg.LOSS.start_steps.classf:
                optimizer = optim.Adam([self.latent], lr=self.cfg.OPTIMIZER.LR_LATENT)

            if i == dynamic_masking_end and self.cfg.N_NOISE_STEPS > 0:
                self.noises = self.models.get_noise(n, trainable=False, random=True)

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True) 
            
            loss = {}
            loss['mse'] = batch_mse_loss(self.image, images_generated, self.masks['mse'])
            
            #mask_I_in = self.models.face_parser(images_generated, mode='mse')
            #print("mask_I_in.shape",mask_I_in.shape)
            
            
            
            #if self.cfg.LOSS.weights.classf > 0 and i >= self.cfg.LOSS.start_steps.classf:
            #    # don't forget convert
            #    images_attr_input = self.convert_attr_input(images_generated)
            #    attr_pred = self.models.classifier(images_attr_input) # tensor([[0.9380, 0.8953, 0.7418]], device='cuda:0', grad_fn=<SigmoidBackward>)
            #    loss['classf'] = kl_divergence(attr_pred, self.target)
                

            #if self.cfg.LOSS.weights.seg > 0 and i >= self.cfg.LOSS.start_steps.seg:
            #    i_dynamic_mask = self.models.face_parser(images_generated, mode='shape')
            #    loss['seg'] = batch_mse_loss(i_dynamic_mask, self.masks['shape'])

            #if self.cfg.SIZE > 0 and i >= self.cfg.LOSS.start_steps.seg:
            #    i_dynamic_mask = self.models.face_parser(images_generated, mode='dynamic')
            #    i_img_portion = get_component_portion(i_dynamic_mask)
            #    loss['size'] = kl_divergence(i_img_portion, target_attr_portion)

            loss_sum = 0
            for term in loss:
                weight = self.cfg.LOSS.weights[term]
                loss_sum += weight * loss[term].sum()
            #print("Losss:",loss_sum)
            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()


            if self.cfg.DYNAMIC_MASKING and dynamic_masking_start <= i < dynamic_masking_end:
                self.masks = self.models.update_image_masks(images_generated, self.original_masks, self.masks)

        print('Finished latent optimization.\n')
        return losses

    def train_noise(self):
        self.noises = [x.requires_grad_(True) for x in self.noises]
        self.latent = self.latent.detach()
        optimizer = optim.Adam(self.noises, lr=self.cfg.OPTIMIZER.LR_NOISE)

        for i in range(self.cfg.N_NOISE_STEPS+1):
            if i % self.cfg.N_ITER_PRINT == 0:
                print(f'Noise optimization {i}/{self.cfg.N_NOISE_STEPS}')

            images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
            
            
            
            loss = {}
            loss['n_loss'] = noise_regularize(self.noises)
            noise_normalize_(self.noises)

            loss['mse'] = batch_mse_loss(self.image, images_generated, self.masks['mse'])
            loss_sum = 0
            for term in loss:
                loss_sum += self.cfg.LOSS.weights[term] * loss[term]

            loss_sum.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Finished noise optimization.\n')


    @torch.no_grad()
    def generate_result(self):
        images_generated = self.models.generator(self.latent, noise=self.noises, to_01=True)
        #blend_mask = self.models.face_parser(images_generated, 'blend')
        result = images_generated 
        print("1111111111111111111111111111111111")
        #result = blend_mask * images_generated + (1 - blend_mask) * self.image
        return result

    def generate_image(self):
        return self.models.generator(self.latent, noise=self.noises, to_01=True)

    @torch.no_grad()
    def generate_comparison(self):
        comparisons = []
        results = self.generate_result()
        for original, res in zip(self.image, results):
            comparisons.append(torch.stack((original, res)))
        comparisons = torch.cat(comparisons, dim=0)
        return comparisons

def batch_mse_loss(input, target, mask=None):
    loss = F.mse_loss(input, target, reduction='none')
    if mask is not None:
        loss = mask * loss
    loss_for_image = torch.mean(loss, dim=(1, 2, 3))
    return loss_for_image

def get_component_portion(mask):
    assert mask.size(2) * mask.size(3)  # todo: delete
    return mask.sum(dim=(2,3)) / (mask.size(2) * mask.size(3))
