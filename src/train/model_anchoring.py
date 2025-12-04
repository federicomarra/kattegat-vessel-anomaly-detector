import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x, lengths):
        # Pack sequence to ignore padding computation
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Encoder outputs: output, (hidden, cell)
        _, (hidden, cell) = self.lstm(packed_x)
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, num_layers, num_ship_types, shiptype_emb_dim, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Ship Type Injection
        self.shiptype_embedding = nn.Embedding(num_ship_types, shiptype_emb_dim)
        
        # --- THE BRIDGE (MODIFIED) ---
        # We now take: Latent (z) + Ship_Emb + Start_Position (input_dim)
        # This "Anchors" the decoding to the correct physical start point.
        total_bridge_input = latent_dim + shiptype_emb_dim + output_dim
        
        self.bridge_hidden = nn.Linear(total_bridge_input, hidden_dim)
        self.bridge_cell = nn.Linear(total_bridge_input, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, z, shiptypes, start_pos):
        # x: (Batch, Seq_Len, Features) -> The input sequence (Teacher Forcing)
        # z: (Batch, Latent_Dim)
        # shiptypes: (Batch)
        # start_pos: (Batch, Features) -> The values at t=0
        
        # 1. Get Ship Embeddings
        ship_emb = self.shiptype_embedding(shiptypes) # (Batch, Emb_Dim)
        
        # 2. Concatenate: Latent + Ship + Start_Pos
        # This creates a context vector that knows WHAT to do (z) and WHERE to start (start_pos)
        combined_features = torch.cat((z, ship_emb, start_pos), dim=1) 
        
        # 3. Project to Initialize Decoder State
        init_hidden = torch.tanh(self.bridge_hidden(combined_features))
        init_cell = torch.tanh(self.bridge_cell(combined_features))
        
        # Repeat for num_layers
        init_hidden = init_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        init_cell = init_cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # 4. LSTM Forward
        output, _ = self.lstm(x, (init_hidden, init_cell))
        
        # 5. Prediction
        prediction = self.fc_out(output)
        
        return prediction

class AIS_LSTM_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, num_ship_types, shiptype_emb_dim, dropout=0.0):
        super(AIS_LSTM_Autoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        
        # Bottleneck Layer
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, num_layers, num_ship_types, shiptype_emb_dim, dropout)
        
    def forward(self, x, lengths, shiptypes):
        # 1. Encode
        hidden, _ = self.encoder(x, lengths)
        
        # 2. Bottleneck
        last_layer_hidden = hidden[-1] 
        z = self.hidden_to_latent(last_layer_hidden) 
        
        # 3. Extract Start Position
        # We take the first element of the sequence (t=0)
        # x shape is (Batch, Max_Len, Features)
        start_pos = x[:, 0, :]
        
        # 4. Decode with Anchoring
        reconstructed = self.decoder(x, z, shiptypes, start_pos)
        
        return reconstructed