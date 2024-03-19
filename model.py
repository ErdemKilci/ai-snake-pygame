import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    """
    Linear Q Network class, representing the neural network model.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the linear Q network.

        Parameters:
            input_size (int): Input size of the network.
            hidden_size (int): Hidden layer size of the network.
            output_size (int): Output size of the network.
        """
        super().__init__()
        # Define the linear layers
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Forward pass through the network
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        """
        Save the model to a file.

        Parameters:
            file_name (str): Name of the file to save.
        """
        # Create model folder if it does not exist
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        # Save the model to the specified file
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    """
    Q Trainer class for training the Q network.
    """

    def __init__(self, model, lr, gamma):
        """
        Initialize the Q trainer.

        Parameters:
            model (nn.Module): Q network model.
            lr (float): Learning rate.
            gamma (float): Discount factor.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Define the optimizer and loss function
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step.

        Parameters:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Action taken.
            reward (torch.Tensor): Reward received.
            next_state (torch.Tensor): Next state.
            done (bool): Flag indicating if the episode is done.
        """
        # Convert inputs to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # If the state is a single sample, add batch dimension
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # Perform forward pass to get predicted Q values
        pred = self.model(state)

        # Compute target Q values
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # Zero gradients, calculate loss, perform backpropagation, and update weights
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
