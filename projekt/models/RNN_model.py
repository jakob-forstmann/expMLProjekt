from models.preprocessing import create_dataset,split_data,build_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim 


# Hyperparameter for the RNN 
EPOCHS = 2
LEARNING_RATE = 0.05
HIDDEN_DIM = 256
OUTPUT_DIM = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def create_data_batches():
    """ loads the songs and 
        splits the transformed songs in batches of size 10
        returns an Iterable for the test respective test Data"""
    spotify_songs = create_dataset()
    X_train,x_test,y_train,y_test = split_data(spotify_songs)
    TF_IDF_Vectorizer = build_model(None)[0]
    transformed_X_train = TF_IDF_Vectorizer.fit_transform(X_train,y_train).todense()
    transformed_x_test = TF_IDF_Vectorizer.transform(x_test).todense()
    train_data = TensorDataset(torch.from_numpy(transformed_X_train), torch.from_numpy(y_train.values))
    test_data = TensorDataset(torch.from_numpy(transformed_x_test), torch.from_numpy(y_test.values))
    return DataLoader(train_data,batch_size=10,shuffle=True),DataLoader(test_data,batch_size=4,shuffle=True)


def train_validate(rnn,training_batch_loader,test_batch_loader):
    """ train the RNN for number of epochs using Batch Learning.
        After training validate and report the training and 
        validation loss"""
    criterion= nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    total_training_loss   = [] 
    total_validation_loss = []
    training_loss = 0.0
    validation_loss = 0.0
    for i in range(0,EPOCHS):
        for train_data,target in iter(training_batch_loader):
            train_data.to(device)
            target.to(device)
            # features have the shape batch x seq length x input_dim
            batch_size = train_data.shape[0]
            hidden_states = rnn.init_hidden(batch_size)
            prediction,hidden_states = rnn(train_data.view(batch_size,1,-1),hidden_states)
            # reshape gold label to batch x seq length x input_dim
            loss  = criterion(prediction,target.view(batch_size,1,-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss+=loss.item()
        total_training_loss.append(training_loss)
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for test_data, target in iter(test_batch_loader):
                batch_size = test_data.shape[0]
                test_data.to(device)
                target.to(device) 
                hidden_states = rnn.init_hidden(batch_size)
                prediction,hidden_states = rnn(test_data.view(batch_size,1,-1),hidden_states)
                loss = criterion(prediction,target.view(batch_size,1,-1))
                validation_loss += loss.item()
        total_validation_loss.append(validation_loss)
    return total_training_loss,total_validation_loss

class RNN(nn.Module):
    def __init__(self, input_size,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.RNN = nn.RNN(input_size,hidden_size,batch_first=True,dtype=torch.float64)
        self.output_layer = nn.Linear(hidden_size,OUTPUT_DIM,dtype=torch.float64)

    def forward(self, x, hidden):
        """ performs a forward pass. Expects the input data 
            to have the shape batch x seq length x input_dim."""
        output,hidden = self.RNN(x,hidden)
        return self.output_layer(output),hidden

    def init_hidden(self,batch_size):
        """initalise all hidden layers with zeros"""
        return torch.zeros(1,batch_size,self.hidden_size,dtype=torch.float64)

def save_results(results,file_name):
    with open(file_name,"w") as f:
        for res in results:
            f.write(f"{str(res)}\n")


if __name__=="__main__":
    train_batch_loader,test_batch_loader = create_data_batches()
    # size of the features after the TF-IDF Transformation
    input_size = 12750
    rnn = RNN(input_size,HIDDEN_DIM)
    if torch.cuda.is_available():
        rnn.to(device)
    train,validation = train_validate(rnn,train_batch_loader,test_batch_loader)
    print(f"train{train} val{validation}")
    save_results(train,"RNN_training")
    save_results(validation,"RNN_validation")

