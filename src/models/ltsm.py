from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,  # This should match the BERT embedding size (768)
        hidden_dim,
        output_dim,
        num_heads=8,
        droupout=0.5,
    ):
        super(LSTM, self).__init__()

        # no embedding layer needed for BERT embeddings
        self.lstm = nn.LSTM(
            input_size=input_size,  # Input size is 768 for BERT
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # multihead attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads
        )

        # hidden layer
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(droupout)
        )

        # output in [0, 1]
        self.output = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, input_sequence):
        # input_sequence shape: (batch_size, sequence_length, embedding_dim)
        hidden_states, _ = self.lstm(input_sequence)  # LSTM expects 3D input

        # permute shape for multihead attention
        hidden_states = hidden_states.permute(1, 0, 2)  # (sequence_length, batch_size, hidden_dim)

        # attention
        context_vector, _ = self.attention(hidden_states, hidden_states, hidden_states)

        # permute back
        context_vector = context_vector.permute(1, 0, 2)  # (batch_size, sequence_length, hidden_dim)
        context_vector = context_vector.mean(dim=1)  # (batch_size, hidden_dim)

        dense_output = self.dense(context_vector)
        output = self.output(dense_output)

        return output
