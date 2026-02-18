import torch 
import torchvision 
from torchvision.transforms import v2 


class intensity_augmentation : 
    def __init__(self , config : dict) : 
        self.config = config 
        self.int_aug = v2.Compose([
            v2.ColorJitter(
                brightness=self.config['data']['augmentation']['intensity']['color_jitter']['brightness'] , 
                contrast=self.config['data']['augmentation']['intensity']['color_jitter']['contrast'] , 
                saturation=self.config['data']['augmentation']['intensity']['color_jitter']['saturation'] , 
                hue = self.config['data']['augmentation']['intensity']['color_jitter']['hue']
            ) , 
            v2.RandomAutocontrast(
                p = self.config['data']['augmentation']['intensity']['RandomAutocontrast']['p']
                ),
            v2.GaussianBlur(
                kernel_size=self.config['data']['augmentation']['intensity']['GaussianBlur']['kernel_size'],
                sigma=self.config['data']['augmentation']['intensity']['GaussianBlur']['sigma']
                ) 
        ]) 

    def forward(self , image , label ) : 
        image , label  = self.int_aug(image , label) 
        return image , label 
        

class geomatric_augmentation : 
    def __init__(self) : 
        NotImplemented 

    def forward(self, image , label) : 
        NotImplemented



class dataset_augmentation : 
    def __init__(self , config ) :  
        self.config = config 
    
    def forward(self , image  , label ) : 
        NotImplemented