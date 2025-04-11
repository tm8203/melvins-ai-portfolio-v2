import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import joblib

# Load dataset (replace with actual dataset loading logic)
def load_data():
    # Dummy dataset example (replace with actual dataset logic)
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = (X[:, 0] > 0.5).astype(int)  # Simple binary target based on first feature
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
def train_logistic_regression(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, 'logistic_regression_model.pkl')  # Save model
    joblib.dump(scaler, 'scaler.pkl')  # Save scaler for inference
    return model, scaler

# Variational Autoencoder (VAE) Definition
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2_mean = nn.Linear(16, latent_dim)
        self.fc2_logvar = nn.Linear(16, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 16)
        self.fc4 = nn.Linear(16, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2_mean(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Train VAE Model
def train_vae(X_train, input_dim, epochs=10, lr=0.001):
    model = VAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, mu, logvar = model(X_train_tensor)
        loss = criterion(recon, X_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'vae_model.pth')  # Save trained VAE model
    return model

if __name__ == "__main__":
    # Load and split data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train and save Logistic Regression Model
    logreg_model, scaler = train_logistic_regression(X_train, y_train)
    
    # Train and save VAE
    vae_model = train_vae(X_train, input_dim=X_train.shape[1])
    
    print("Model training completed and saved.")
