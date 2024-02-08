import pandas as pd
import numpy as np
import torch

from model import Model

def preprocess_data(data, is_train=True):

    selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if is_train:
        y = data['Survived']
    '''
    https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    Pandas use the copy-on-write mechanism, we should avoid "chained indexing", so we use .copy() to avoid it.
    '''
    data = data[selected_features].copy()

    data['Age'] = data['Age'].fillna(data['Age'].mean()).astype(float)
    data['Fare'] = data['Fare'].fillna(data['Fare'].mean())
    data['Embarked'] = data['Embarked'].fillna('S')

    data['Sex'] = data['Sex'].map({'female': 0, 'male': 1}).astype(int)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    data = torch.tensor(data.values, dtype=torch.float32)

    if is_train:
        return data, torch.tensor(y.values, dtype=torch.float32)
    else:
        return data

def train(model, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    pred = model(x_train)
    loss = model.loss(pred, y_train)
    loss.backward()
    optimizer.step()
    y_pre = (pred > 0.5).int().squeeze()
    acc = (y_pre == y_train).float().mean()
    return loss.item(), acc

def test(model, x_test):
    model.eval()
    with torch.no_grad():
        pred = model(x_test)
        pred = (pred > 0.5).int().squeeze()
    return pred


def main():

    # Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    lr = 0.001

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load data
    print("Data Preprocessing...")
    train_dir = "./titanic/train.csv"
    test_dir = "./titanic/test.csv"
    submission_dir = "./titanic/gender_submission.csv"
    train_df = pd.read_csv(train_dir)
    test_df = pd.read_csv(test_dir)
    submission = pd.read_csv(submission_dir)

    # print(train_df.head()) # print first 5 rows of train_df
    # print(test_df.head())

    # print(train_df.dtypes)
    '''
    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object
    '''

    x_train, y_train = preprocess_data(train_df, is_train=True)
    x_test = preprocess_data(test_df, is_train=False)

    # print(x_train.shape) # torch.Size([891, 7])
    # print(y_train.shape) # torch.Size([891])
    # print(x_test.shape) # torch.Size([418, 7])

    x_train, y_train, x_test = x_train.to(device), y_train.to(device), x_test.to(device)

    # Train model
    print("Training model...")
    model = Model(input_dim=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(epochs+1):
        loss, acc = train(model, optimizer, x_train, y_train)
        print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(i, epochs, loss, acc))

    # Test model
    print("Testing model...")
    pred = test(model, x_test)
    submission['Survived'] = pred.cpu().numpy()
    submission.to_csv('submission.csv', index=False)
    print("Done! Please check submission.csv")

if __name__ == "__main__":
    main()