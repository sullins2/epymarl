import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch as th

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        # self.bn1 = nn.BatchNorm1d(args.hidden_dim) 
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        # self.bn2 = nn.BatchNorm1d(args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        self.lr = nn.ReLU()

        # Weight initialization
        init.xavier_uniform_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0.1)
        init.xavier_uniform_(self.fc2.weight)
        init.constant_(self.fc2.bias, 0.1)

        print("NAMED PARAMETERS")
        print(self.named_parameters)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state, t_env):
        # x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        # if self.args.use_rnn:
        #     h = self.rnn(x, h_in)
        # else:
        #     h = F.relu(self.rnn(x))
        # q = self.fc2(h)

        # x = F.relu(self.bn1(self.fc1(inputs)))  # Apply batch normalization after fc1
        # h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        # if self.args.use_rnn:
        #     h = self.rnn(x, h_in)
        # else:
        #     h = F.relu(self.bn2(self.rnn(x)))  # Apply batch normalization after rnn
        # q = self.fc2(h)

        # if t_env > 23000: # and t_env % 1000 == 0:
        #   # print("TEST MODE: ", test_mode)
        #   print("INPUT")
        #   print(inputs)
        #   # print("Hidden state")
        #   # print(hidden_state)
        #   # print("X:")
        #   # print(x)
        #   # print("Q")
        #   # print(q)
        #   # print("H")
        #   # print(h)
        #   for name, param in self.named_parameters():
        #     if param.grad is not None:
        #       print('Gradient of {} ({})'.format(name, param.shape))
        #       print(param.grad)

        x = self.lr(self.fc1(inputs))  # Apply batch normalization after fc1
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = self.lr(self.rnn(x))  # Apply batch normalization after rnn
        q = self.fc2(h)

        ##########
        # L2 Regularization
        l2_lambda = 0.001  # Regularization parameter
        l2_reg = th.tensor(0.)  # Initialize regularization term

        # Calculate regularization term for each parameter
        for param in self.parameters():
            l2_reg += th.norm(param)

        # Add regularization term to the loss
        l2_loss = l2_lambda * l2_reg
        loss = q + l2_loss
        return loss, h

        # return q, h
