

from keras.models import load_model
from PIL import Image
import numpy as np
import torch
import os
from TransKey import TransKeyR, Bottleneck
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

TransPoseR = TransKeyR


class Normalize(object):
    def __init__(self, mean, std, H, W):
        self.mean = mean
        self.std = std
        self.H = H
        self.W = W

    def __call__(self, image):
        image_copy = np.copy(image)
        if isinstance(image_copy, np.ndarray):
            print(image_copy.shape)
        else:
            print("image_copy non è un array NumPy.")
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        image_copy = (image_copy - self.mean) / self.std

        return image_copy


class ToTensor(object):
    def __call__(self, image):
        if len(image.shape) == 2:  # se l'immagine è in scala di grigi
            image = image.reshape(image.shape[0], image.shape[1], 1)
        image = image.transpose((2, 0, 1))
        #output = {'image': torch.from_numpy(image)}
        return torch.from_numpy(image)
    

class Rescale(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image):
    
        img = cv2.resize(image, (self.W, self.H))
        return img
    

def denormalize_keypoints(keypoints_normalized, width, height):
     # Moltiplica per le dimensioni dell'immagine per denormalizzare
    keypoints_original_x = keypoints_normalized[:, 0] * (width - 1)
    keypoints_original_y = keypoints_normalized[:, 1] * (height - 1)

    keypoints_original = np.stack([keypoints_original_x, keypoints_original_y], axis=-1)
    return keypoints_original

def make_prediction(model, image_dir, mean, std, H, W):

    image = Image.open(image_dir)

    larghezzaO, altezzaO = image.size
    image_np = np.array(image)

    data_transform =transforms.Compose(
    [
        Rescale(H, W), 
        Normalize(mean, std, H, W),
        ToTensor()
    ])

    image_t = data_transform(image_np).unsqueeze(0)

    image2 = image_t.float()

    outputs, _, _= model(image2)   

    predicted_keypoints = outputs[0].detach().numpy()
    predicted_keypoints = denormalize_keypoints(predicted_keypoints, larghezzaO, altezzaO)

    
    # Disegna i keypoints sull'immagine originale
    for i, kp in enumerate(predicted_keypoints):
        x, y = int(kp[0]), int(kp[1])
        raggio = int((larghezzaO + altezzaO) / 2 * 0.008)
        cv2.circle(image_np, (int(x), int(y)), raggio, (255, 0, 0), -1)

        dimensione_label = ((larghezzaO + altezzaO) / 2 * 0.0007)
        label = f"{i+1}"  
        cv2.putText(image_np, label, (x+8, y-10), cv2.FONT_HERSHEY_SIMPLEX, dimensione_label, (0, 255, 255), 2)

    # Salva l'immagine modificata
    #modified_image_path = 'static/images/modified' + os.path.basename(image_dir)
    #cv2.imwrite(modified_image_path, image_np)

    return image_np, predicted_keypoints, larghezzaO, altezzaO


def main():
   
    BN_MOMENTUM = 0.1

    H = 256
    W = 256

    resnet_spec = {
                    50: (Bottleneck, [3, 4, 6, 3]),
                101: (Bottleneck, [3, 4, 23, 3]),
                152: (Bottleneck, [3, 8, 36, 3])}


    block_class, layers = resnet_spec[50]


    # Carico il modello TransKeyR
    model = TransKeyR(block_class, layers, BN_MOMENTUM, W, H)

    # Carico i pesi del modello
    model = torch.load('best_saved_model.pth', map_location=torch.device('cpu'))
    model.eval()

    mean = 0.485 * 255
    std = 0.229 * 255


    file_path = input("Inserisci il percorso dell'immagine che si vuole predire: ")

    try:
        with open(file_path, 'rb') as file:
            content = file.read()
    except FileNotFoundError:
        print("L'immagine non è stata trovata nel percorso specificato.")

    img_mod, coordinates, widthO, heightO = make_prediction(model, file_path, mean, std, H, W)
    coordinates_str = '\n'.join([f"Key {i+1}: {x:.3f} {y:.3f}" for i, (x,y) in enumerate(coordinates)])
    print(f"\nCoordinate predette scalate secondo la dimensione originale dell'immagine ({widthO} ,{heightO}):")
    print(coordinates_str)
    
    plt.imshow(img_mod.astype('uint8'))  
    plt.axis('off')  
    plt.show()

    
    
    

    

if __name__ == '__main__':
    main()
    
