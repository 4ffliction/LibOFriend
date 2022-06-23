from optparse import Values
from pickletools import optimize
import tez
import pandas as pd
from sklearn import model_selection
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
import numpy as np

class BookDataset:
    def __init__(self, Name, Rating=None):
        self.Name= Name
        self.Rating= Rating
    
    def __len__(self):
        return len(self.Name)

    def __getitem__(self, item):
        Name= self.Name[item]
        Rating= self.Rating[item]
    

        return {"Name": torch.tensor(Name, dtype=torch.long),
        "Rating": torch.tensor(Rating, dtype=torch.float)}



class RecSysModel(tez.Model):
    def __init__(self, num_Name):
        super().__init__()
        self.Book_embed= nn.Embedding(num_Name, 32)
        self.out= nn.Linear(64, 1)
        self.step_scheduler_after="epoch"

    def fetch_optimizer(self):
        opt= torch.optim.Adam(self.parameters(), ir=1e-3)
        return opt

    def fetch_scheduler(self):
        sch= torch.optim.lr.scheduler.StepLR(self.optimizer, step_size= 3, gamma=0.7)
        return sch
            

    def monitor_metrics(self, output, rating):
        output= output.detach().cpu().numpy()
        rating= rating.detach().cpu().numpy()
        return{
            'rmse': np.sqrt(metrics.mean_squared_error(rating, output))

            }

    def forward(self, Name, Rating= None):
        user_embeds= self.Name_embed(Name)
        output=torch.cat(user_embeds)
        output= self.out(output)
                
        loss= nn.MSELoss()(output, Rating.view(-1,1))
        calc_metrics= self.monitor_metrics(output, Rating.view(-1,1))
        return output, loss, calc_metrics


        
def train():
    df= pd.read_csv("../input/Raw Dataset - Sheet1.csv")
    lbl_Name= preprocessing.LabelEncoder()

    df.Name- lbl_Name.fit_transform(df.Name.values)
    df_train, df_valid=model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.rating.Values
    )
    train_dataset= BookDataset(
        df_train.Name.values, df_train.Rating.values
    )

    valid_dataset= BookDataset(
        df_valid.Name.values, df_valid.Rating.values
    )

    model= RecSysModel(num_Name=len(lbl_Name.classes_))
    model.fit(
        train_dataset, valid_dataset, train_bs= 1024,
        valid_bs=1024, fp16= True
    )
if __name__== "__main__":
    train()



 
