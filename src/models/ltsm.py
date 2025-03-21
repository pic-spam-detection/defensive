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

        self.projection = nn.Linear(input_size, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # multihead attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=num_heads, batch_first=True
        )

        # hidden layer
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(droupout)
        )

        # output in [0, 1]
        self.output = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Sigmoid())

    def forward(self, input_sequence):
        projected = self.projection(input_sequence)  # (batch_size, hidden_dim)

        if projected.dim() == 2:
            projected = projected.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        hidden_states, _ = self.lstm(projected)

        # attention
        context_vector, _ = self.attention(hidden_states, hidden_states, hidden_states)

        context_vector = context_vector.mean(dim=1)  # (batch_size, hidden_dim)

        dense_output = self.dense(context_vector)
        output = self.output(dense_output)

        return output
