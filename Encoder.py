import numpy
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import tanh
import math
import utils
from ActorCriticNetwork import Encoder, Decoder


class Encoder(nn.Module):
    """
    Encoder of TSP-Net
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 n_nodes,
                 n_rnn_layers):
        """
        Initialise Encoder
        :param int input_dim: Number of input dimensions
        :param int embedding_dim: Number of embbeding dimensions
        :param int hidden_dim: Number of hidden units of the RNN
        :param int n_layers: Number of RNN layers
        :param int n_nodes: Number of nodes in the TSP
        """
        super(Encoder, self).__init__()
        self.n_rnn_layers = n_rnn_layers
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes

        self.embedding = nn.Linear(input_dim, embedding_dim)

        self.g_embedding = nn.Linear(embedding_dim, hidden_dim)
        self.g_embedding1 = nn.Linear(hidden_dim, hidden_dim)
        self.g_embedding2 = nn.Linear(hidden_dim, hidden_dim)

        self.rnn0 = nn.LSTM(input_size=embedding_dim, #128
                            hidden_size=hidden_dim, #128
                            num_layers=n_rnn_layers, #1
                            batch_first=True)

        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_dim,
                           num_layers=n_rnn_layers,
                           batch_first=True)

        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

        self.rnn0_reversed = nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_dim,
                                     num_layers=n_rnn_layers,
                                     batch_first=True)

        self.rnn_reversed = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=n_rnn_layers,
                                    batch_first=True)

        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_b = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input, hidden=None):
        """
        Encoder: Forward-pass

        :param Tensor input: Graph inputs (bs, n_nodes, 2)
        :param Tensor hidden: hidden vectors passed as inputs from t-1
        """

        batch_size = input.size(0)

        edges = utils.batch_pair_squared_dist(input, input)
        edges.requires_grad = False

        # embedding shared across all nodes
        embedded_input = self.embedding(input)

        if hidden is None:
            h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_rnn_layers,
                                                          batch_size,
                                                          self.hidden_dim)
            c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_rnn_layers,
                                                          batch_size,
                                                          self.hidden_dim)
        else:
            h0, c0 = hidden
            h0 = h0.detach()
            c0 = h0.detach()

            h0 = h0.unsqueeze(0).repeat(self.n_rnn_layers, 1, 1)
            c0 = c0.unsqueeze(0).repeat(self.n_rnn_layers, 1, 1)

        g_embedding = embedded_input \
            + F.relu(torch.bmm(edges, self.g_embedding(embedded_input)))
        g_embedding = g_embedding \
            + F.relu(torch.bmm(edges, self.g_embedding1(g_embedding)))
        g_embedding = g_embedding \
            + F.relu(torch.bmm(edges, self.g_embedding2(g_embedding)))

        rnn_input = g_embedding
        rnn_input_reversed = torch.flip(g_embedding, [1])

        # first RNN reads the last node on the input
        rnn0_input = rnn_input[:, -1, :].unsqueeze(1)
        self.rnn0.flatten_parameters()
        _, (h0, c0) = self.rnn0(rnn0_input, (h0, c0))
        # second RNN reads the sequence of nodes
        self.rnn.flatten_parameters()
        s_out, s_hidden = self.rnn(rnn_input, (h0, c0))

        # first RNN reads the last node on the input
        rnn0_input_reversed = rnn_input_reversed[:, -1, :].unsqueeze(1)
        self.rnn0_reversed.flatten_parameters()
        _, (h0_r, c0_r) = self.rnn0_reversed(rnn0_input_reversed)
        # second RNN reads the sequence of nodes
        self.rnn_reversed.flatten_parameters()
        s_out_reversed, s_hidden_reversed = self.rnn_reversed(rnn_input_reversed,
                                                              (h0_r, c0_r))

        s_out = tanh(self.W_f(s_out)
                     + self.W_b(torch.flip(s_out_reversed, [1])))

        s_hidden = (s_hidden[0]+s_hidden_reversed[0],
                    s_hidden[1]+s_hidden_reversed[1])

        return s_out, s_hidden, _, g_embedding

    encoder = Encoder(2, 128, 128, 20, 1)

    points = [[0.9026783735, 0.0872014179], [0.2688188533, 0.8870915955], [0.9571323399, 0.8528540973],
              [0.9644893261, 0.1674329695], [0.1830417743, 0.6097890516], [0.8496648693, 0.2917820319],
              [0.6703750016, 0.090733078], [0.3654335632, 0.9946665667], [0.5607386437, 0.944230069],
              [0.1815313745, 0.5975951437], [0.3636787984, 0.3961268719], [0.8977224691, 0.0012632063],
              [0.6508970548, 0.7949156352], [0.8119892313, 0.3103225925], [0.7641752744, 0.1801678016],
              [0.9661493989, 0.5020117279], [0.4361717029, 0.9128186484], [0.4377802724, 0.7792351437],
              [0.160172209, 0.7148703235], [0.9055019075, 0.7435757641]]

    input_lstm = points

    inputs = numpy.array(input_lstm)
    input_arr = []
    for i in range(20):
        input_arr.append(inputs)
    inputs_tensor = torch.from_numpy(numpy.array(input_arr)).float()
    hidden = None
    #print(inputs_tensor)
    #print(input_arr)
    s_out, s_hidden, _, g_embedding = encoder(inputs_tensor, hidden)
    #print(s_out)
    #print("###################")
    #print(s_hidden)

    enc_h = (s_hidden[0][-1], s_hidden[1][-1])

    #print(enc_h)


    decoder = Decoder(
        128,
        128,
        2
    )
    actions = None
    probs, pts, log_probs_pts, entropies = decoder(enc_h,  # last layer of lstm
                                                          s_out,  # lstm output
                                                          s_out,  # lstm output
                                                          actions,  # None
                                                          g_embedding,  # gcn layer
                                                          enc_h)  # last layer of star lstm


    print(probs)
