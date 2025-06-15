import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # batch_first=True for (batch, seq, feature)

    def forward(self, x):
        # x shape: (batch_size, look_back_steps, 1)
        # output: (batch_size, look_back_steps, hidden_size)
        # hidden, cell: (num_layers, batch_size, hidden_size) - final hidden/cell state for each layer
        output, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_horizon):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_hidden, encoder_cell):

        batch_size = encoder_hidden.size(1) # hidden_state: (num_layers, batch_size, hidden_size)
        decoder_input = torch.zeros(batch_size, self.forecast_horizon, self.output_size).to(encoder_hidden.device)

        # Initialize decoder's hidden and cell states with encoder's final states
        output, (hidden, cell) = self.lstm(decoder_input, (encoder_hidden, encoder_cell))
        # output shape: (batch_size, forecast_horizon, hidden_size)

        predictions = self.fc(output) # predictions shape: (batch_size, forecast_horizon, output_size)
        return predictions

class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, look_back_steps, forecast_horizon):
        super(Seq2SeqAutoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(output_size, hidden_size, num_layers, output_size, forecast_horizon) # Input to decoder LSTM is output_size

    def forward(self, x):
        encoder_hidden, encoder_cell = self.encoder(x)
        predictions = self.decoder(encoder_hidden, encoder_cell)
        return predictions