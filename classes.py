import tensorflow.keras as k
import numpy as np


class Agent(object):
    def __init__(self, state_size, action_size, BATCH_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.gamma = 0.95
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.discount_factor = .9

        self.actor, self.critic = self.build_models()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

    def build_models(self):
        state = k.layers.Input(batch_shape=(None, self.state_size))
        actor_input = k.layers.Dense(32, activation='relu',
                                     kernel_initializer='he_uniform')(state)
        mu = k.layers.Dense(self.action_size, activation='tanh',
                            kernel_initializer='he_uniform')(actor_input)
        sigma = k.layers.Dense(self.action_size, activation='softplus',
                               kernel_initializer='he_uniform')(actor_input)

        critic_input = k.layers.Dense(32, activation='relu',
                                      kernel_initializer='he_uniform')(state)
        state_value = k.layers.Dense(1, kernel_initializer='he_uniform')(critic_input)

        actor = k.Model(inputs=state, outputs=(mu, sigma))
        critic = k.Model(inputs=state, outputs=state_value)

        return actor, critic

    def actor_optimizer(self):
        epsilon = 1e-8
        advantages = k.backend.placeholder(shape=(None, 1))

        mu, sigma_sq = self.actor.output
        action = mu + np.random.normal(self.action_size) * sigma_sq
        pre_sum = -0.5 * ((((action - mu) ** 2) / ((sigma_sq + epsilon) ** 2)) + 2 *
                          k.backend.log(sigma_sq) + k.backend.log(2 * np.pi))
        eligibility = k.backend.log(pre_sum + 1e-10) * k.backend.stop_gradient(advantages)
        actor_loss = -k.backend.sum(eligibility)

        opt = k.optimizers.Adam(learning_rate=self.actor_lr, beta_1=0.9, epsilon=1e-1)
        updates = opt.get_updates(actor_loss, self.actor.trainable_weights)
        train = k.backend.function([self.actor.input, advantages], [actor_loss], updates=updates)
        return train

    def critic_optimizer(self):
        discounted_reward = k.backend.placeholder(shape=(None, 1))
        value = self.critic.output

        loss = k.backend.mean(k.backend.square(discounted_reward - value))

        opt = k.optimizers.Adam(lr=self.critic_lr)

        updates = opt.get_updates(loss, self.critic.trainable_weights)
        train = k.backend.function([self.critic.input, discounted_reward], [loss], updates=updates)
        return train

    def act(self, state):
        mu, sigma_sq = self.actor.predict(np.reshape(state, [1, self.state_size]))
        epsilon = np.random.randn(self.action_size)
        action = mu + np.sqrt(sigma_sq) * epsilon
        action = np.clip(action, -2, 2)
        return int(np.amax(action))

    def train(self, state, next_state, reward, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(np.reshape(state, [1, self.state_size]))
        next_value = self.critic.predict(np.reshape(next_state, [1, self.state_size]))

        if done:
            advantages[0] = reward - value
            target[0][0] = reward
        else:
            advantages[0] = reward + self.discount_factor * next_value - value
            target[0][0] = reward + self.discount_factor * next_value

        self.optimizer[0]([np.reshape(state, [1, self.state_size]), advantages])
        self.optimizer[1]([np.reshape(state, [1, self.state_size]), target])

