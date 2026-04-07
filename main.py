import numpy as np
import pandas as pd

# 1. FEATURE SCALING

def standardize(X):
 mean = np.mean(X, axis=0)
 std = np.std(X, axis=0)
 return (X - mean) / std, mean, std


# 2. MSE LOSS
def mse_loss(y_true, y_pred):
 return np.mean((y_true - y_pred) ** 2)


# 3. FORWARD PASS
def forward(X, w, b):
 return np.dot(X, w) + b



# 4. BACKWARD PASS
# Ağırlıklar ve bias için gradyanları hesapla
def compute_gradients(X, y_true, y_pred):
 n = len(y_true)
 error = y_pred - y_true

 dw = (2 / n) * np.dot(X.T, error)
 db = (2 / n) * np.sum(error)

 return dw, db


# 5. SGD OPTIMIZER
def sgd_update(w, b, dw, db, lr):
 w = w - lr * dw
 b = b - lr * db
 return w, b


# 6. ADAM OPTIMIZER
class AdamOptimizer:
 def __init__(self, shape, lr=0.001):
     self.lr = lr
     self.beta1 = 0.9
     self.beta2 = 0.999
     self.epsilon = 1e-8
 
     self.m_w = np.zeros(shape)
     self.v_w = np.zeros(shape)
     self.m_b = 0 
     self.v_b = 0
     self.t = 0

 def update(self, w, b, dw, db):
     self.t += 1
     self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
     self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
     self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
     self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
    
    
     m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
     v_w_hat = self.v_w / (1 - self.beta2 ** self.t)

     m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
     v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

     w = w - self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
     b = b - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

     return  w, b


# 7. VERİ SETİNİ YÜKLE
# Oyuncu bilgileri, piyasa değerleri ve performans verileri
print("Dosyalar okunuyor, lütfen bekleyin...")

players = pd.read_csv("players.csv")
top_5_ids = ['GB1', 'ES1', 'IT1', 'L1', 'FR1']
players = players[players['current_club_domestic_competition_id'].isin(top_5_ids)]

appearances = pd.read_csv("appearances.csv", usecols=['player_id', 'goals', 'assists', 'minutes_played'])
appearances = appearances[appearances['player_id'].isin(players['player_id'])]

valuations = pd.read_csv("player_valuations.csv")
valuations = valuations.sort_values("date").drop_duplicates("player_id", keep="last")

print("Veriler birleştiriliyor...")

performance = appearances.groupby("player_id").agg({
    "goals": "sum",
    "assists": "sum",
    "minutes_played": "sum"
}).reset_index()

df = pd.merge(players, valuations, on="player_id")
df = pd.merge(df, performance, on="player_id")

print(f"Başarılı! Toplam {len(df)} kaliteli oyuncu verisi ile eğitime geçiliyor.")


# 8. FEATURE ENGINEERING
# Yaş, gol katkısı gibi yeni özellikler oluştur
df["goal_contribution"] = df["goals"] + df["assists"]


# 9. MODEL VERİSİ
# Eksik değerleri temizle ve sadece sayısal özellikleri kullanarak model verisi oluştur
print("Mevcut Sütunlar:", df.columns.tolist())

df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
df['age'] = 2026 - df['date_of_birth'].dt.year
df["goal_contribution"] = df["goals"] + df["assists"]

# Sadece Top 5 ligde oynayan ve değeri olan oyuncular
df_model = df[df["market_value_in_eur_x"] > 500000][[
    "age", "goals", "assists", "minutes_played", "goal_contribution", "market_value_in_eur_x"
]].dropna()



# X ve y değişkenlerini ata
X = df_model.drop(columns=["market_value_in_eur_x"]).values
y = df_model["market_value_in_eur_x"].values



# 10. LOG TRANSFORM
# Piyasa değerleri genellikle büyük ve dağılımı bozuk olabilir, bu yüzden log transform uygulayarak daha normal bir dağılım elde edebiliriz.
y = np.log1p(y)


# 11. FEATURE SCALING
# Modelin daha hızlı ve stabil öğrenmesi için özellikleri standartlaştır
X_scaled, x_mean, x_std = standardize(X)



# 12. MODEL PARAMETRELERİ
# Ağırlıklar ve bias'ı sıfırla
n_features = X_scaled.shape[1]
w = np.zeros(n_features)
b = 0

epochs = 1000
optimizer = AdamOptimizer(shape=n_features, lr=0.05)


# 13. TRAINING
# Modeli eğitmek için epoch'lar boyunca forward, loss hesaplama, backward ve update adımlarını uygula
for epoch in range(epochs):
    
# Forward
    
    y_pred = forward(X_scaled, w, b)

 # Loss
    loss = mse_loss(y, y_pred)

 # Backward
    dw, db = compute_gradients(X_scaled, y, y_pred)

# Update
    w, b = optimizer.update(w, b, dw, db)

if (epoch + 1) % 20 == 0:
 print(f"Epoch {epoch+1} - Loss: {loss:.4f}")


# 14. MODEL SONUÇ
# Eğitim tamamlandıktan sonra ağırlıklar ve bias'ı yazdır
print("\nModel tamamlandı.")
print("Ağırlıklar:", w)
print("Bias:", b)

# 15. YENİ OYUNCU TAHMİNİ
# Yeni bir oyuncunun yaş, gol, asist, dakika ve gol katkısı bilgilerini kullanarak piyasa değerini tahmin et
# [age, goals, assists, minutes_played, goal_contribution]
X_new = np.array([[25, 10, 5, 2000, 15]])

X_new_scaled = (X_new - x_mean) / x_std

y_pred = forward(X_new_scaled, w, b)

# log geri çevir
y_real = np.expm1(y_pred)

print("\nTahmini piyasa değeri (€):", y_real)