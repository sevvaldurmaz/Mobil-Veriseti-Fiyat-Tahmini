from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import re
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Tüm satır ve sütunları görebilmek için:
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

#Veriyi excel dosyasıdan okuttum.
dataset_path = r"C:\Users\sevva\Desktop\Mobiles\MobilesDataset\Mobiles Dataset_2025.xlsx"
dataFrame = pd.read_excel(dataset_path)

#Model Name sütununu 2'ye parçalayıp model adı ayrı hafıza(gb) ayrı sütun olacak şekilde ayırdım.
def modelname_parcala(text):
    text = str(text)
    parts = text.split()
    memory = parts[-1] if parts[-1].endswith('GB') else None
    model = ' '.join(parts[:-1]) if memory else ' '.join(parts)
    return pd.Series([model, memory])

#Model ve Memory sütunlarını oluşturup, Model Name sütununu kalıcı olarak siliyorum.
dataFrame[['Model','Memory']] = dataFrame['Model Name'].apply(modelname_parcala)
dataFrame.drop(columns=['Model Name'], inplace=True)

#Model ve Memory sütunları veri setinin sonuna eklendiği için onları tekrardan eski yerlerine getiriyorum.
#2.Sütun Model 3.Sütun Memory olacak.
sutunlar = dataFrame.columns.tolist()  # Tüm sütunları listeye aldım.
model_sutunu = sutunlar[-2]
sutunlar.remove(model_sutunu)
sutunlar.insert(1, model_sutunu)
sonsutun = sutunlar[-1]
sutunlar.remove(sonsutun)
sutunlar.insert(2, sonsutun)
dataFrame = dataFrame[sutunlar]

#Memory sütunu "GB" içerdiği için sadece sayısal değerleri alıyoruz.
def memory_int_donustur(memo):
    if isinstance(memo, str):
        match = re.search(r'\d+', memo)
        return int(match.group()) if match else None
    return None

dataFrame['Memory'] = dataFrame['Memory'].apply(memory_int_donustur)

#Memory sütununda bulunan non değerleri ortalama ile doldurdum.
dataFrame['Memory'] = dataFrame['Memory'].replace('', np.nan)
dataFrame['Memory'] = dataFrame['Memory'].astype(float)
mean_value = dataFrame['Memory'].mean()
dataFrame['Memory'] = dataFrame['Memory'].fillna(mean_value)
dataFrame["Memory"] = dataFrame["Memory"].astype(float).astype(int)

dataFrame['Mobile Weight'] = dataFrame['Mobile Weight'].str.replace(r'\D', '', regex=True).astype(int)
dataFrame['RAM'] = dataFrame['RAM'].str.replace(r'\D', '', regex=True).astype(int)
dataFrame['Battery Capacity'] = dataFrame['Battery Capacity'].str.replace(r'\D', '', regex=True).astype(int)
dataFrame['Screen Size'] = dataFrame['Screen Size'].str.replace(r'\D', '', regex=True).astype(int)
dataFrame['Launched Price (Pakistan)'] = dataFrame['Launched Price (Pakistan)'].astype(str).str.replace(r'\D', '', regex=True).replace('', pd.NA).astype('Int64')
dataFrame['Launched Price (India)'] = dataFrame['Launched Price (India)'].str.replace(r'\D', '', regex=True).replace('', pd.NA).astype('Int64')
dataFrame['Launched Price (China)'] = dataFrame['Launched Price (China)'].str.replace(r'\D', '', regex=True).replace('', pd.NA).astype('Int64')
dataFrame['Launched Price (USA)'] = dataFrame['Launched Price (USA)'].str.replace(r'\D', '', regex=True).replace('', pd.NA).astype('Int64')
dataFrame['Launched Price (Dubai)'] = dataFrame['Launched Price (Dubai)'].str.replace(r'\D', '', regex=True).replace('', pd.NA).astype('Int64')

#Bu fonksiyon on ve arka kamera sütunlarındaki gereksiz ifadeleri silip düzenlememize yarar.
#MP yazılarını kaldırdım, Parantez içleri(kategorik veri) kaldırdım. /4 ifadelerini kaldırdım. ve + olan tüm ifadeleri toplayıp sonucu yazdırdım.
def camera_sutun_temizle(value):
    if pd.isna(value):
        return None
    value = str(value).lower()
    value = value.replace("mp", "")
    value = re.sub(r'\(.*?\)', '', value)
    value = value.split('/')[0]
    value = re.sub(r'4k.*', '', value)
    numbers = re.findall(r'\d+', value)
    total = sum(map(int, numbers)) if numbers else None
    return total

dataFrame["Front Camera"] = dataFrame["Front Camera"].apply(camera_sutun_temizle)
dataFrame["Back Camera"] = dataFrame["Back Camera"].apply(camera_sutun_temizle)

#Model, Company Name ve Processor sütunu da yine kategorik veri ieçriyor label encoding uyguladım.
leb = LabelEncoder()
dataFrame['Model'] = leb.fit_transform(dataFrame['Model'])
dataFrame['Processor'] = leb.fit_transform(dataFrame['Processor'])
dataFrame['Company Name'] = leb.fit_transform(dataFrame['Company Name'])

# Korelasyon matrisini hesaplayıp ısı haritasını çizdirdim.
correlation_matrix = dataFrame.corr()
plt.figure(figsize=(12, 8))
sbn.heatmap(correlation_matrix, annot=True, cmap="coolwarm",fmt=".2f", linewidths=0.5)
plt.title('Korelasyon Isı Haritası')
plt.show()

print(dataFrame)
#Veri setinin describe'ni aldım.
print(dataFrame.describe())
#Non değerleri kontrol ettim.
print(dataFrame.isnull().sum())
#veri tiplerini kontrol ettim.
print(dataFrame.info())

#hedef değişkenlerimi belirledim. 5 adet hedef değişkenim var.
x = dataFrame.drop(columns=['Launched Price (Pakistan)', 'Launched Price (India)', 'Launched Price (China)', 'Launched Price (USA)', 'Launched Price (Dubai)'])
y = dataFrame[['Launched Price (Pakistan)', 'Launched Price (India)', 'Launched Price (China)', 'Launched Price (USA)', 'Launched Price (Dubai)']]
y = np.log1p(y)

#Eğitim ve test olarak ayırdım.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=16)

#Min-max scaler ile normalizasyon yaptım.
scaler = MinMaxScaler()
print(scaler.fit(x_train))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#Kullanacağım modelleri beklirledim. Birden fazla hedef değişkenim olduğundan MultiOutputRegressor kullandım.
gbm_model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=15)
multi_model = MultiOutputRegressor(gbm_model)

#Modeli eğittim.
multi_model.fit(x_train, y_train)

#Tahminler üzerinde deneme yaptım.
tahminler = multi_model.predict(x_test)
tahminler = np.expm1(tahminler)
y_test = np.expm1(y_test)

#Model Sonuçları
print("Ortalama R² skoru:", round(r2_score(y_test, tahminler) * 100, 2), "%")
print("Ortalama MAE:", round(mean_absolute_error(y_test, tahminler), 2))
print("Ortalama MSE:", round(mean_squared_error(y_test, tahminler), 2))
print("Ortalama RMSE:", round(np.sqrt(mean_squared_error(y_test, tahminler)), 2))


