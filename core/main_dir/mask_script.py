import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pretty_errors

# Function to create and return model
def create_mask_detector_model(pretrained=False):
    model = resnet18(weights=pretrained)
    model.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=1),
        nn.Sigmoid()
    )
    return model

class Train():
    def __init__(self, batch_size=32, val_batch_size=8, learning_rate=1e-3, start_new=True, liveplot=True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_path = os.getcwd()
        self.display = liveplot

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.train_dataset = datasets.ImageFolder(root=os.path.join(self.curr_path, 'data', 'train'), transform=train_transform)
        self.val_dataset = datasets.ImageFolder(root=os.path.join(self.curr_path, 'data', 'validation'), transform=val_transform)

        print(f"\nFound {len(self.train_dataset)} images for training")
        print(f"Found {len(self.val_dataset)} images for validating")

        if torch.cuda.is_available():
            print(f"Using GPU : {torch.cuda.get_device_name(0)}")
        else:
            print(f"No GPU available, using CPU.")

        # Move datasets to the device
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

        # Define your PyTorch model (make sure it's designed to run on the specified device)
        if start_new:
            self.model = create_mask_detector_model(pretrained=True).to(self.device)
            self.message = "new mask_detector"
        else:
            self.model = create_mask_detector_model(pretrained=False).to(self.device)
            self.message = "existing mask_detector"
            try:
                self.model.load_state_dict(torch.load(os.path.join(self.curr_path, 'models', 'mask_detector.pth'), map_location=self.device))
            except:
                raise Exception(f"Model mask_detector failed to load")

        # Define loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-7, weight_decay=False, amsgrad=False)

    def train(self, num_epochs) -> None:

        print(f"\nTraining {self.message} model for {num_epochs} epoch(s)...\n")

        # Initialize regularization strengths
        l1_lambda = 0.01
        l2_lambda = 0.01

        if self.display:
            fig, axes = plt.subplots(1, 2, figsize=(9, 5))
            ax1, ax2 = axes

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):

            # Use tqdm for progress bar during training
            self.model.train()
            with tqdm(total=len(self.train_loader), bar_format=f'Epoch {epoch + 1}/{num_epochs} | Train      '+'|{bar:30}{r_bar}', unit=' batch(s)') as pbar:
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                for inputs, labels in self.train_loader:
                    inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                    # Ensure labels are one-dimensional
                    labels = labels.unsqueeze(-1)

                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # Add L1 and L2 regularization
                    l1_reg = torch.tensor(0., requires_grad=True)
                    l2_reg = torch.tensor(0., requires_grad=True)

                    for name, param in self.model.named_parameters():
                        if 'weight' in name:
                            l1_reg = l1_reg + torch.norm(param, 1)
                            l2_reg = l2_reg + torch.norm(param, 2)

                    loss_with_l1_l2 = loss + l1_lambda * l1_reg + l2_lambda * l2_reg

                    # Backward pass and optimization
                    loss_with_l1_l2.backward()
                    self.optimizer.step()

                    # Update metrics
                    total_loss += loss_with_l1_l2.item()
                    predicted = (outputs.data > 0.5).float()
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    # Update progress bar
                    avg_loss = total_loss / total_samples
                    accuracy = correct_predictions / total_samples
                    pbar.set_postfix({'loss': avg_loss, 'accuracy': accuracy})
                    pbar.update(1)

                train_losses.append(total_loss / total_samples)
                train_accuracies.append(correct_predictions / total_samples)

                pbar.close()

            # Validation
            self.model.eval()
            with torch.no_grad():
                # Use tqdm for progress bar during validation
                with tqdm(total=len(self.val_loader), bar_format=f'        '+' '*len(str(epoch+1))+' '*len(str(num_epochs))+'| Validation '+'|{bar:30}{r_bar}', unit=' batch(s)') as pbar:
                    val_total_loss = 0.0
                    val_correct_predictions = 0
                    val_total_samples = 0

                    for inputs, labels in self.val_loader:
                        inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                        # Ensure labels are one-dimensional
                        labels = labels.unsqueeze(-1)

                        # Forward pass for validation
                        outputs = self.model(inputs)
                        val_loss = self.criterion(outputs, labels)

                        # Update metrics for validation
                        val_total_loss += val_loss.item()
                        val_predicted = (outputs.data > 0.5).float()
                        val_total_samples += labels.size(0)
                        val_correct_predictions += (val_predicted == labels).sum().item()

                        # Update progress bar
                        val_avg_loss = val_total_loss / val_total_samples
                        val_accuracy = val_correct_predictions / val_total_samples
                        pbar.set_postfix({'val_loss': val_avg_loss, 'val_accuracy': val_accuracy})
                        pbar.update(1)

                    val_losses.append(val_total_loss / val_total_samples)
                    val_accuracies.append(val_correct_predictions / val_total_samples)

                    pbar.close()

            if self.display:
                # Update live plots
                ax1.clear()
                ax2.clear()

                # Adjust x-axis values
                x_values = list(range(1, epoch + 2))

                ax1.plot(x_values, train_losses, label='Training Loss')
                ax1.plot(x_values, val_losses, label='Validation Loss')
                ax1.set_title('Losses')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()

                ax2.plot(x_values, train_accuracies, label='Training Accuracy')
                ax2.plot(x_values, val_accuracies, label='Validation Accuracy')
                ax2.set_title('Accuracies')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()

                plt.suptitle('Live Training Stats\nmask_detector', fontsize=14)
                plt.pause(0.1)


            # Save the trained model
            self.criterion = nn.BCELoss()
            self.optimizer = optim.Adam(self.model.parameters())
            torch.save(self.model.state_dict(), os.path.join(self.curr_path, 'models', 'mask_detector.pth'))
            print()
        print(f'Training complete\n')
        if self.display:
            plt.show()

class Test():
    def __init__(self, batch_size=8) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_path = os.getcwd()

        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        self.test_dataset = datasets.ImageFolder(root=os.path.join(self.curr_path, 'data', 'test'), transform=test_transform)

        print(f"\nFound {len(self.test_dataset)} images for testing")

        if torch.cuda.is_available():
            print(f"Using GPU : {torch.cuda.get_device_name(0)}")
        else:
            print(f"No GPU available, using CPU.")

        # Move dataset to the device
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        # Define your PyTorch model (make sure it's designed to run on the specified device)
        self.model = create_mask_detector_model(pretrained=False).to(self.device)
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.curr_path, 'models', 'mask_detector.pth'), map_location=self.device))
        except:
            raise Exception(f"Model mask_detector failed to load")

        self.criterion = nn.BCELoss()

    def test(self, con_matrix=True) -> None:

        print(f"\nEvaluating mask_detector model...\n")

        # Use tqdm for progress bar during testing
        self.model.eval()
        with torch.no_grad():
            all_labels = []
            all_predictions = []

            with tqdm(total=len(self.test_loader), bar_format='Evaluation '+'|{bar:30}{r_bar}', unit=' batch(s)') as pbar:
                total_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                    labels = labels.unsqueeze(-1)

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # Update metrics
                    total_loss += loss.item()
                    predicted = (outputs.data > 0.5).float()
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

                    # Collect labels and predictions for later evaluation
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())

                    # Update progress bar
                    avg_loss = total_loss / total_samples
                    accuracy = correct_predictions / total_samples
                    pbar.set_postfix({'loss': avg_loss, 'accuracy': accuracy})
                    pbar.update(1)

                pbar.close()

            # Convert lists to numpy arrays
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)

        print(f'\nEvaluation complete')

        # Generate and print classification report and confusion matrix
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['Absent', 'Present']))

        if con_matrix:
            con_mat = confusion_matrix(all_labels, all_predictions)
            display = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=['Absent', 'Present'])
            display.plot()
            plt.title('Confusion Matrix')
            plt.show()

class Predict():
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.curr_path = os.getcwd()

        if torch.cuda.is_available():
            print(f"\nUsing GPU : {torch.cuda.get_device_name(0)}")
        else:
            print(f"\nNo GPU available, using CPU.")

        # Define your PyTorch model (make sure it's designed to run on the specified device)
        self.model = create_mask_detector_model(pretrained=False).to(self.device)
        try:
            self.model.load_state_dict(torch.load(os.path.join(self.curr_path, 'models', 'mask_detector.pth'), map_location=self.device))
        except:
            raise Exception(f"Model mask_detector failed to load")

    def predict(self, image_path, display_image=True):
        predict_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = predict_transform(image).unsqueeze(0).to(self.device)

        # Perform prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor).cpu()
            output = np.array(output)[0][0]
            output = float(output)

            if output < 0.5:
                prediction = 'Absent'
                confidence = str(round(1 - output*100, 4)) + '%'
            else:
                prediction = 'Present'
                confidence = str(round(output*100, 4)) + '%'
        
        # Print result
        print(f"\nPredicted given image \"{image_path}\" as \"{prediction}\" with {confidence} confidence.\n")

        # Display the original image and the prediction
        if display_image:
            plt.imshow(image)
            plt.title(f'Prediction: {prediction}\nConfidence: {confidence}', fontsize=16)
            plt.xlabel(image_path, fontsize=13)
            plt.xticks([])
            plt.yticks([])  
            plt.show()

if __name__ == '__main__':
    trainer = Train(batch_size=16, val_batch_size=8, learning_rate=1e-3, start_new=True)
    trainer.train(num_epochs=50)

    tester = Test(batch_size=8)
    tester.test()

    predictor = Predict()
    image_path = 'data_small/test/Present/test-present-0.jpg'
    predictor.predict(image_path, display_image=True)
