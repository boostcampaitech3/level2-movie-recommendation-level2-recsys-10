import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        total_input_dim = int(sum(input_dims)) # n_user + n_movie + n_year

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)
        self.linear = nn.Linear(in_features  = 18, out_features = embedding_dim, bias = False) # 장르를 리니어로

        self.embedding = nn.Embedding(total_input_dim, embedding_dim) 
        self.embedding_dim = (len(input_dims)+1) * embedding_dim # 장르를 추가해서 1더한것 더이상 추가하면 큰일남

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim)) # embedding_dim에 장르의 크기를 더해주어야 함
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim)) 
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input)
        x , genre = x[:, :-18], x[:, -18:].to(torch.float32)
        embed_x = self.embedding(x)  # [50,3,10]
        embeded_genre = self.linear(genre).unsqueeze(1) # [50,1,10]
        input = torch.cat([embed_x, embeded_genre], dim = 1)  # [50,4,10]

        # fm_y = (self.bias + torch.sum(self.fc(x), dim=1) + self.linear(genre).squeeze(-1)).unsqueeze(1) # [50,1]
        fm_y = self.bias + torch.sum(self.fc(x), dim=1) + torch.sum(embeded_genre, dim = 2) # [50,1]
        # print(fm_y, fm_y.size())
        square_of_sum = torch.sum(input, dim=1) ** 2  # [50,10]       
        sum_of_square = torch.sum(input ** 2, dim=1)  # [50,10]          
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True) #[50,1] 
        return fm_y
    
    def mlp(self, x):
        x , genre = x[:, :-18], x[:, -18:].to(torch.float32)
        embed_x = self.embedding(x)
        embeded_genre = self.linear(genre).unsqueeze(1)
        input = torch.cat([embed_x, embeded_genre], dim = 1).view(-1, self.embedding_dim)  # [50,4,10] -> [50, 40]

        mlp_y = self.mlp_layers(input)
        return mlp_y

    def forward(self, x):
        #fm component
        fm_y = self.fm(x).squeeze(1) # [50,1] -> [50]
        #deep component
        mlp_y = self.mlp(x).squeeze(1) #  [50,1] -> [50]
        y = torch.sigmoid(fm_y + mlp_y)
        return y


class DeepFM_renew(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM_renew, self).__init__()
        total_input_dim = int(sum(input_dims)) # n_user3만 + n_genre700 + n_year100 + n_writer_director640(일단 디렉터 없으면 작가 그도 없으면 기타1)

        # Fm component의 constant bias term과 1차 bias term
        # 모두 embedding으로 교체(장르 : Linear -> Embedding)
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)

        self.embedding = nn.Embedding(total_input_dim, embedding_dim) 
        self.embedding_dim = (len(input_dims)) * embedding_dim # 필드 갯수(4) * embedding_dim

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim)) # embedding_dim에 장르의 크기를 더해주어야 함
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim)) 
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)  # [50,4,10]

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2  # [50,10]       
        sum_of_square = torch.sum(embed_x ** 2, dim=1)  # [50,10]        

        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True) #[50,1] 
        return fm_y
    
    def mlp(self, x):
        embed_x = self.embedding(x) # [b, 4, e_d]
        input = embed_x.view(-1, self.embedding_dim) #[b, 4*e_d]

        mlp_y = self.mlp_layers(input)
        return mlp_y

    def forward(self, x):
        #fm component
        fm_y = self.fm(x).squeeze(1) # [50,1] -> [50]
        #deep component
        mlp_y = self.mlp(x).squeeze(1) #  [50,1] -> [50]
        y = torch.sigmoid(fm_y + mlp_y)
        return y
