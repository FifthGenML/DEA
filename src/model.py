import requests
import base64
import io
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from src.utils import score, perturb, crossover, mutate, visualize_perturbation_difference
import torch.nn.functional as F
from IPython.display import display
import matplotlib.pyplot as plt

class WolfDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image), self.labels[idx]

class SurrogateModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval() 
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.wolf_class_idx = 269  # ImageNet index for timber wolf etc
    
    def predict(self, images):
        self.model.eval()
        probs = []
        
        with torch.no_grad():
            for img in images:
                # Convert numpy image to tensor and preprocess
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                

                outputs = self.model(img_tensor)
                probs_all = F.softmax(outputs, dim=1)
                
    
                wolf_prob = probs_all[0][self.wolf_class_idx].item()
                probs.append([wolf_prob, 1 - wolf_prob])  
        
        return np.array(probs)

class DEAttack:
    def __init__(self, original_image, population_size=50, mutation_factor=0.8, crossover_rate=0.7):
        # Resize original image to model input size
        if isinstance(original_image, np.ndarray):
            original_image = Image.fromarray(original_image)
        original_image = original_image.resize((224, 224), Image.BILINEAR)
        self.original_image = np.array(original_image)
        
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.surrogate_model = SurrogateModel()
        self.population = self.initialize_population()
    
    def initialize_population(self):
        population = []
        total_pixels = 224 * 224  
        
        for _ in range(self.population_size):
            img = self.original_image.copy()

            perturb(img, magnitude=0.8, pixels=total_pixels // 4)
            population.append(img)
        return population
    
    def evaluate_fitness(self, population):
        # Get predictions from surrogate model
        predictions = self.surrogate_model.predict(population)
        wolf_probs = predictions[:, 0]  
        
        num_wolf = np.sum(wolf_probs > 0.5)
        avg_prob = np.mean(wolf_probs)
        min_prob = np.min(wolf_probs)
        max_prob = np.max(wolf_probs)
        
        print(f"MobileNetV2 Predictions:")
        print(f"  - Images predicted as wolf: {num_wolf}/{len(population)} ({num_wolf/len(population)*100:.1f}%)")
        print(f"  - Wolf probability: min={min_prob:.3f}, avg={avg_prob:.3f}, max={max_prob:.3f}")
        
        return wolf_probs
    
    def evolve(self, population, fitness_scores):
        new_population = []
        total_pixels = 224 * 224
        
        for i in range(len(population)):

            candidates = list(range(len(population)))
            candidates.remove(i)
            a, b, c = np.random.choice(candidates, 3, replace=False)
            

            trial = population[a].copy()
            diff = population[b].astype(np.float32) - population[c].astype(np.float32)
            trial = trial.astype(np.float32) + self.mutation_factor * diff
            np.clip(trial, 0, 255, out=trial)
            trial = trial.astype(np.uint8)
            
            # Additional targeted mutation for exploration
            if np.random.random() < self.crossover_rate:

                mutate(trial, 0.4, total_pixels // 10)
            
            # Selection step
            trial_pred = self.surrogate_model.predict([trial])[0]
            if trial_pred[0] < fitness_scores[i]:  # If trial is better
                new_population.append(trial)
            else:
                new_population.append(population[i])
        
        return new_population
    
    def attack(self, max_generations=100, api_check_interval=1):
        best_attack = None
        best_fitness = float('inf')
        
        print("\nStarting Differential Evolution Attack...")
        print(f"Initial API prediction: {get_prediction_probs(self.original_image)}")
        
        # old_style_img = self.original_image.copy()
        # perturb(old_style_img, magnitude=0.8, pixels=50000) 
        
        # new_style_img = self.population[0].copy() 
        
        # fig = visualize_perturbation_difference(self.original_image, old_style_img, new_style_img)
        # display(fig)
        # plt.close(fig)
        
        for gen in range(max_generations):
            print(f"\nGeneration {gen + 1}/{max_generations}")
            print("-" * 40)
            

            fitness_scores = self.evaluate_fitness(self.population)
            

            min_idx = np.argmin(fitness_scores)
            if fitness_scores[min_idx] < best_fitness:
                best_fitness = fitness_scores[min_idx]
                best_attack = self.population[min_idx].copy()
                print(f" New best fitness: {best_fitness:.3f}")
                

                display(Image.fromarray(best_attack))
            
            # Check with real API after set interval
            if gen % api_check_interval == 0 and best_attack is not None:
                print("\n Checking best candidate with API...")
                api_pred = get_prediction_probs(best_attack)
                print(f"API Prediction: {api_pred}")
                if api_pred != "timber wolf":  # Success!
                    print(f"\n Attack succeeded at generation {gen + 1}!")
                    print(f"Final API prediction: {api_pred}")
                    return best_attack
            
            # Evolve!
            self.population = self.evolve(self.population, fitness_scores)
        
        print("\n Maximum generations reached without success")
        return best_attack

def get_prediction_probs(image_np):
    image = Image.fromarray(image_np.astype(np.uint8))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = score(img_str)
    print(max(response['output']))
    pred_class = max(response["output"])[1]
    return pred_class