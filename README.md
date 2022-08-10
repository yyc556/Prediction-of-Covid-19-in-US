# Covid-19在美國之確診數預測
<br>深度學習概論-英家慶 課程期末專題
<br>組員：陳彥妤、鄭巧翎、黃翊瑄
* 資料集
<br>[Novel Corona Virus 2019 Dataset | Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)
* 使用PyTorch
* 模型建立
<br>Simple RNN, LSTM, GRU, CNN+LSTM

## 資料預處理
1. 累加確診人數
```
import pandas as pd
time_series_data = pd.DataFrame()
df = pd.DataFrame(df[df.columns[11:]].sum(), columns=['confirmed'])
time_series_data = pd.concat([time_series_data,df], axis=1)
time_series_data.index = pd.to_datetime(time_series_data.index, format='%m/%d/%y')
time_series_data.reset_index(inplace=True)
```
<img src="https://github.com/yyc556/prediction-of-Covid-19-in-USA/blob/main/images/cumulative%20confirmed%20cases.png">

2. MinMaxScaler
利用MinMaxScaler將數據縮到[0,1]之間
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(time_series_data['confirmed'], axis=1))

time_series_data['confirmed'] = scaler.transform(np.expand_dims(time_series_data['confirmed'], axis=1))
```
3. 切割資料集
```
split = round(0.8*len(time_series_data))
train_data = time_series_data['confirmed'][:split]
test_data = time_series_data['confirmed'][split:]
```
4. 建立序列
利用前五天的數據預測第六天之確診人數
```
def create_sequences(data, previous):
  X, y = [], []
  for i in range(len(data)-previous-1):
      x = data[i:(i+previous)]
      X.append(x)
      y.append(data[i+previous])
  X = np.array(X)
  y = np.array(y)

  return X, y

previous = 5  # 利用前五天預測第六天的確診人數
X_train, y_train = create_sequences(train_data.to_numpy(), previous)
X_test, y_test = create_sequences(test_data.to_numpy(), previous)
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
```
5. 統一資料維度
```
X_train = X_train.unsqueeze(2)
X_test = X_test.unsqueeze(2)
y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)
```

## 模型建立
```
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
```
* Simple RNN
```
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_class):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_class)
    
  def forward(self, x):
    out, _ = self.rnn(x)
    out = out[:, -1, :]
    out = self.fc(out)
    return out
```
* LSTM
```

class LSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_class):
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_class)
    
  def forward(self, x):
    h0 = Variable(torch.zeros(num_layers, x.size(0), hidden_size))
    c0 = Variable(torch.zeros(num_layers, x.size(0), hidden_size))
    out, _ = self.lstm(x, (h0, c0))
    out = out[:, -1, :]
    out = self.fc(out)
    return out
```
* GRU
```
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.linear(out)
        return out
```
* CNN+LSTM
```
class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.conv = nn.Sequential(
            nn.Conv1d(5, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=1, kernel_size=1)
        )
        self.LSTM1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.LSTM2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.conv(x)
        out, _ = self.LSTM1(out)
        out, _ = self.LSTM2(out)
        out = out[:, -1, :]
        out = self.linear(out)
        return out
```
模型架構整理
<br><img src="https://github.com/yyc556/prediction-of-Covid-19-in-USA/blob/main/images/model%20structure.png" width=80%>

## 模型訓練
<br>Loss Function：MSE
<br>Optimizer：Adam
<br>Learning Rate：0.001
<br>Epoch：200
```
def train_model(model, X_train, y_train, X_test=None, y_test=None):
  loss_fn = nn.MSELoss()
  optimizer = opt.Adam(model.parameters(), lr = 0.001)
  num_epoches = 200

  train_loss_hist = np.zeros(num_epoches)
  test_loss_hist = np.zeros(num_epoches)

  for epoch in range(num_epoches):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if X_test is not None:
      with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred, y_test)
      test_loss_hist[epoch] = test_loss.data

      if (epoch+1)%10 == 0:
        print('Epoch: %d, Train Loss: %.4f, Test Loss: %.4f' %  (epoch+1, loss.data, test_loss.data))
    else:
      if (epoch+1)%10 == 0:
        print('Epoch: %d, Loss: %.4f' %  (epoch+1, loss.data))
    train_loss_hist[epoch] = loss.data 
    
  return y_pred, train_loss_hist, test_loss_hist  
```

## 訓練結果
* LOSS
```
def loss_plot(model, X_train, y_train, X_test, y_test):
  y_pred, train_loss_hist, test_loss_hist = train_model(model, X_train, y_train, X_test, y_test)
  plt.plot(train_loss_hist, label='Train Loss')
  plt.plot(test_loss_hist, label='Test Loss')
  plt.legend()
```
<img src="https://github.com/yyc556/prediction-of-Covid-19-in-USA/blob/main/images/loss%20compare.png" width=85%>

* Prediction
```
def result_plot(model, train_data, X_test, y_test):
  with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_pred = y_test_pred.flatten()

  train_data = scaler.inverse_transform(train_data.values.reshape(-1,1))
  y_test = scaler.inverse_transform(y_test.reshape(-1,1))
  y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1,1))

  plt.plot(time_series_data['confirmed'].index[:len(train_data)],
          train_data,
          label = 'Historical Cumulative Cases')
  plt.plot(time_series_data['confirmed'].index[len(train_data):len(train_data)+len(y_test_pred)],
          y_test,
          label = 'Real Cumulative Cases')
  plt.plot(time_series_data['confirmed'].index[len(train_data):len(train_data)+len(y_test_pred)],
          y_test_pred,
          label = 'Predicted Cumulative Cases')
  plt.legend()
```
<img src="https://github.com/yyc556/prediction-of-Covid-19-in-USA/blob/main/images/prediction%20compare.png" width=85%>

## 預測未來
利用資料集原始數據及預測出的未來30天確診數
```
def predict(model, num_prediction):
  with torch.no_grad():
    T = 5
    predict_list = time_series_data['confirmed'][-T:]
    num_prediction = num_prediction
    for _ in range(num_prediction):
      x = predict_list[-T:]
      x = np.array(x)
      x = torch.tensor(x).float()
      x = x.reshape((1,T,1))
      pred = model(x)
      predict_list = np.append(predict_list, pred)
  predict_list = predict_list[T-1:]
  predict_list = scaler.inverse_transform(predict_list.reshape(-1,1)).astype(int)
  last_date = time_series_data['index'].values[-1]
  predict_dates = np.array(pd.date_range(last_date, periods=num_prediction+1))
  return predict_dates, predict_list
```
```
# 將所有的資料集數值反標準化
real_comfirmed = scaler.inverse_transform(time_series_data['confirmed'].values.reshape(-1,1))
```
```
rnn_predict_dates, rnn_predict_list = predict(rnn, 30)
lstm_predict_dates, lstm_predict_list = predict(lstm, 30)
gru_predict_dates, gru_predict_list = predict(gru, 30)
convlstm_predict_dates, convlstm_predict_list = predict(convlstm, 30)
```
```
# 作圖
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
plt.plot(time_series_data['index'], real_comfirmed)
plt.plot(rnn_predict_dates, rnn_predict_list)
plt.plot(lstm_predict_dates, lstm_predict_list)
plt.plot(gru_predict_dates, gru_predict_list)
plt.plot(convlstm_predict_dates, convlstm_predict_list)
plt.title('Cumulaive Confirmed Cases')
plt.xlabel('Date')
plt.ylabel('Cumulaive Confirmed Cases')
plt.legend(['Real','Forecast_RNN','Forecast_LSTM','Forecast_GRU','Forecast_ConvLSTM'],loc = 'lower right')
plt.show()
```
<img src="https://github.com/yyc556/prediction-of-Covid-19-in-USA/blob/main/images/future%20prediction.png">

## 結論
1. 從測試結果來看，LSTM的結果是最好的。LSTM所預測出來的結果也最符合實際狀況。
2. 預測不準確的可能原因：只把確診數丟進去訓練，沒有其他環境因素可以學習
3. 各模型比較
<img src="https://github.com/yyc556/prediction-of-Covid-19-in-USA/blob/main/images/conclusison.png" width=80%>

## 參考資料
[Using Conv + LSTM to predict Covid 19](https://www.kaggle.com/derrelldsouza/using-conv-lstm-to-predict-covid-19#4.Predictive-modelling-using-CNN-+-Bi-Directional-LSTM)
<br>[US Covid Case Prediction - Keras RNN](https://www.kaggle.com/tammamkhan/us-covid-case-predicton-keras-rnn#Plot-Model-Predictions)
