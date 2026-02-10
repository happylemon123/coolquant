import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    A custom implementation of a Graph Attention Layer (GAT).
    
    Mathematical Formulation:
    e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
    alpha_ij = softmax_j(e_ij)
    h'_i = sigma(sum_j alpha_ij * Wh_j)
    
    Why from scratch? To demonstrate understanding of the 
    Attention Mechanism beyond just importing 'torch_geometric'.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.6, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Learnable weight matrix W
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Learnable attention vector 'a'
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            h: Input node features [N, in_features]
            adj: Adjacency matrix [N, N] (0 or 1, or weighted)
            
        Returns:
            h_prime: Output node features [N, out_features]
        """
        # 1. Linear Transformation
        Wh = torch.mm(h, self.W) # [N, out_features]
        
        # 2. Attention Mechanism
        # We need to compute attention for every pair of nodes.
        # Check this neat broadcasting trick to avoid loops:
        N = Wh.size()[0]
        
        # Explain this to the user in the interview:
        # We repeat the matrix to create N x N pairs
        a_input = self._prepare_attentional_mechanism_input(Wh) # [N, N, 2*out_features]
        
        # Compute e_ij (unnormalized attention scores)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # [N, N]
        
        # 3. Masked Attention
        # We only pay attention to neighbors (where adj > 0)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 4. Softmax normalization
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 5. Aggregation
        h_prime = torch.matmul(attention, Wh) # [N, out]
        
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Constructs the correlation input for the attention mechanism.
        Returns a tensor of shape [N, N, 2*out_features]
        concatenating all pairs (Wh_i, Wh_j).
        """
        N = Wh.size()[0]
        # [N, 1, out] -> [N, N, out]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # [N*N, 2*out]
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        # Reshape to [N, N, 2*out]
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

class ST_GNN(nn.Module):
    """
    Spatiotemporal GNN Wrapper.
    Combines LSTM (Temporal) with GAT (Spatial/Graph).
    """
    def __init__(self, n_features, n_hidden, n_classes, dropout):
        super(ST_GNN, self).__init__()
        
        # Temporal: LSTM to process time series per node
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, batch_first=True)
        
        # Spatial: Graph Attention to process relationships between nodes
        self.gat = GraphAttentionLayer(n_hidden, n_hidden, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(n_hidden, n_classes)
        
    def forward(self, x, adj):
        """
        x: [Batch, Nodes, Time, Features]
        """
        # Simplification for single-snapshot training:
        # x is [Nodes, Time, Features]
        
        # 1. Temporal Encoding (LSTM)
        # Output: [Nodes, Time, Hidden] -> We take last time step: [Nodes, Hidden]
        lstm_out, _ = self.lstm(x) 
        node_embeddings = lstm_out[:, -1, :] 
        
        # 2. Spatial Aggregation (GAT)
        spatial_out = self.gat(node_embeddings, adj)
        
        # 3. Prediction
        out = self.fc(spatial_out)
        return out
