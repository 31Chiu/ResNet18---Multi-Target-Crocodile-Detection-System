import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets                    # Use our own customized dataset module
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import os                                           # For file path operations
import logging
from tqdm import tqdm                               # To show a progress bar

logging.basicConfig(
    level=logging.INFO,                                                 # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',                 # Define the log message format
    handlers=[
        logging.FileHandler('knn_classifier_resnet18.log', encoding='utf-8'),       # Log messages to a file named 'knn_classifier_resnet18.log'
        logging.StreamHandler()                                                     # Also log messages to the console
    ]
)

# Step 1: Definitions and Preparations
def get_device():
    # Gets the available compute device (GPU or CPU).
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pretrained_resnet18(model_path='best_resnet18_model.pth', num_classes=10):
    """
    Loads a pre-trained ResNet18 model and modifies it into a feature extractor.
    We remove the final classification layer, so the model's output is the feature vector.
    """
    # Load the model structure
    model = models.resnet18(weights=None)   # weights=None as we will load our own

    # Get the number of input features for the classifier
    num_ftrs = model.fc.in_features
    # # The Sequential structure here needs to match your training script
    # model.fc = nn.Sequential(
    #     nn.Dropout(0.5),
    #     nn.Linear(num_ftrs, num_classes)
    # )
    # Structure that perfectly matches the training time
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Load the entire saved file, which is a dictionary (checkpoint) containing various places of data.
    checkpoint = torch.load(model_path, map_location=get_device())

    # Extract only the model's state dictionary (the weight) from the checkpoint.
    model.load_state_dict(checkpoint['model_state_dict'])

    # Remove the final fully-connected layer to make it a feature extractor
    # We use nn.Identity() as a placeholder that does nothing
    model.fc = nn.Identity()

    device = get_device()
    model = model.to(device)
    model.eval()    # Set the model to evaluation mode
    return model

# Step 2: Feature Extraction
def extract_features(dataloader, model):
    # Iterates through the dataset and use the ResNet18 model to extract features from all images.
    features_list = []
    labels_list = []
    device = get_device()

    with torch.no_grad():   # No need to calculate gradients, saves memory and computation
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            # The model's output is now a batch of feature vectors
            feature_batch = model(images)

            # Move features and labels to CPU and convert to NumPy arrays
            features_list.append(feature_batch.cpu().numpy())
            labels_list.append(labels.numpy())

        # Concatenate lists of arrays into single large NumPy arrays
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        return features, labels
    
# Step 3: Main Execution Flow
def main():
    # Main function to orchestrate all steps.
    logging.info("Starting KNN classification process...")

    # Get the base directory of the script
    # Instead of using a relative path like './', we construct an absolute path based on the current script location.
    # This ensures that no matter what folder you run the script from, it will always find the correct file.
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # !! IMPORTANT !!
    # Define the path to your dataset and the number of classes you have.
    # The dataset should be structured.
    DATASET_PATH = os.path.join(base_dir, 'dataset')                                        # Adjust this path to your dataset location
    NUM_CLASSES = 2
    MODEL_PATH = os.path.join(base_dir, 'resnet18_checkpoint', 'best_resnet18_model.pth')   # Path to the model checkpoint

    logging.info(f'Dataset Path: {DATASET_PATH}')
    logging.info(f'Model Path: {MODEL_PATH}')
    logging.info(f'Number of Classes: {NUM_CLASSES}')

    # 1. Load the model
    print("Loading ResNet18 feature extractor model...")
    logging.info("Loading ResNet18 feature extrator model...")
    resnet_feature_extractor = load_pretrained_resnet18(model_path=MODEL_PATH, num_classes=NUM_CLASSES)

    # 2. Prepare the dataset
    # Define the same transformations used during training
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Resize(680),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading custom dataset...")
    logging.info("Loading custom dataset...")
    # Load the train and validate sets using ImageFolder
    train_dataset = datasets.ImageFolder(root=f'{DATASET_PATH}/Training', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{DATASET_PATH}/Validation', transform=transform)
    logging.info(f'Found {len(train_dataset)} training images and {len(test_dataset)} validation images.')

    # Create DataLoaders
    # Adjust batch_size based on your hardware's capability
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    # 3. Extract features
    print("Extracting features from the training set...")
    logging.info("Extracting features from the training set...")
    train_features, train_labels = extract_features(train_loader, resnet_feature_extractor)

    print("Extracting feature from the test set...")
    logging.info("Extracting feature from the test set...")
    test_features, test_labels = extract_features(test_loader, resnet_feature_extractor)

    print(f'Feature extraction complete! Train features shape: {train_features.shape}, Test features shape: {test_features.shape}')
    logging.info(f'Feature extraction complete! Train features shape: {train_features.shape}, Test features shape: {test_features.shape}')

    # 4. Train and evaluate the KNN classifier
    print("\n--- Training and Evaluating KNN Classifier ---")
    logging.info("--- Training and Evaluating KNN Classifier ---")

    # Define a value for K, a key parameter for the KNN algorithm
    k_value = 5
    print(f'Using K = {k_value}')
    logging.info(f'Using K = {k_value}')

    # Create an instance of the KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=k_value, n_jobs=-1) # n_jobs=-1 uses all available CPU cores

    print("Training the KNN classifier...")
    logging.info("Training the KNN classifier...")
    # "Training" for KNN is simply memorizing the training data
    knn.fit(train_features, train_labels)

    print("Making predictions with the KNN classifier...")
    logging.info("Making predictions with the KNN classifier...")
    # Make predictions on the test set
    predictions = knn.predict(test_features)

    # 5. Calculate and display the accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print("------------------------------------------")
    logging.info("------------------------------------------")
    print(f'KNN classifier accuracy on the test set: {accuracy * 100:.2f}%')
    logging.info(f'FINAL RESULT: KNN classifier accuracy on the test set: {accuracy * 100:.2f}%')
    print("------------------------------------------")
    logging.info("------------------------------------------")
    logging.info("KNN classification process finished.")

if __name__ == '__main__':
    main()
