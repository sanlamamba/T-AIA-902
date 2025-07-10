import numpy as np
from collections import deque
import random
from .base import BaseAgent


class DQNAgent(BaseAgent):
    def __init__(
        self,
        n_actions,
        n_states,
        alpha=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=10000,
        batch_size=32,
    ):
        super().__init__(n_actions, n_states)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.training_step = 0

        # Improved network architecture
        self.weights1 = np.random.randn(n_states, 256) * np.sqrt(2.0 / n_states)
        self.bias1 = np.zeros(256)
        self.weights2 = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
        self.bias2 = np.zeros(128)
        self.weights3 = np.random.randn(128, 64) * np.sqrt(2.0 / 128)
        self.bias3 = np.zeros(64)
        self.weights4 = np.random.randn(64, n_actions) * np.sqrt(2.0 / 64)
        self.bias4 = np.zeros(n_actions)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _forward(self, state):
        self.z1 = np.dot(state, self.weights1) + self.bias1
        self.a1 = self._relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self._relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self._relu(self.z3)
        self.z4 = np.dot(self.a3, self.weights4) + self.bias4
        return self.z4

    def _backward(self, state, action, error):
        # Output layer gradients
        dz4 = np.zeros_like(self.z4)
        dz4[action] = error
        
        dw4 = np.outer(self.a3, dz4)
        db4 = dz4
        
        # Hidden layer 3 gradients
        dz3 = np.dot(dz4, self.weights4.T) * self._relu_derivative(self.z3)
        dw3 = np.outer(self.a2, dz3)
        db3 = dz3
        
        # Hidden layer 2 gradients
        dz2 = np.dot(dz3, self.weights3.T) * self._relu_derivative(self.z2)
        dw2 = np.outer(self.a1, dz2)
        db2 = dz2
        
        # Hidden layer 1 gradients
        dz1 = np.dot(dz2, self.weights2.T) * self._relu_derivative(self.z1)
        dw1 = np.outer(state, dz1)
        db1 = dz1
        
        # Update weights and biases
        self.weights4 += self.alpha * dw4
        self.bias4 += self.alpha * db4
        self.weights3 += self.alpha * dw3
        self.bias3 += self.alpha * db3
        self.weights2 += self.alpha * dw2
        self.bias2 += self.alpha * db2
        self.weights1 += self.alpha * dw1
        self.bias1 += self.alpha * db1

    def get_action(self, state, explore=True):
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_one_hot = np.zeros(self.n_states)
        state_one_hot[state] = 1
        q_values = self._forward(state_one_hot)
        return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        total_error = 0

        for state, action, reward, next_state, done in batch:
            state_one_hot = np.zeros(self.n_states)
            state_one_hot[state] = 1
            next_state_one_hot = np.zeros(self.n_states)
            next_state_one_hot[next_state] = 1

            current_q = self._forward(state_one_hot)

            if done:
                target = reward
            else:
                next_q = self._forward(next_state_one_hot)
                target = reward + self.gamma * np.max(next_q)

            error = target - current_q[action]
            total_error += abs(error)
            self._backward(state_one_hot, action, error)

        self.training_errors.append(total_error / self.batch_size)
        self.training_step += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
