import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import shap

# 1. Veri setini yükleme ve ön işleme.

df = pd.read_csv("turkish_ecommerce_reviews.csv")

"""print("Veriden ilk 5 satır:")
print(df.head())

print("\nEksik Veri Kontrolü:")
print(df.isnull().sum())

print("\nSınıf Dağılımı:")
print(df['Rating (Star)'].value_counts()) 

print('Satır sayısı:', df.shape[0])
print('Sütun sayısı:', df.shape[1])
print('Sütun isimleri:', df.columns.tolist())

print(df.info())
print(df.describe())"""

# 2. Metin temizleme ve sınıflandırma.

df = df[df["Rating (Star)"] != 3].copy()
df["Sentiment"] = df["Rating (Star)"].apply(lambda x: 1 if x > 3 else 0)

def clean_text(text):
    text = str(text).lower()  
    text = re.sub(r'[^a-zçğıöşü\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["Clean_Review"] = df["Review"].apply(clean_text)

print("Yeni Sınıf Dağılımı (0: Olumsuz, 1: Olumlu):")
print(df['Sentiment'].value_counts())

# 3. Metinleri tokenleştirme ve stopwords kaldırma.

nltk.download('stopwords')
dur_kelimeleri = set(stopwords.words('turkish'))

dur_kelimeleri.update(['bir', 'için', 'çok', 'gibi', 'kadar', 'daha', 've', 
                       'ile', 'ama', 'da', 'de', 'bu', 'o', 'ben', 'sen', 'biz', 
                       'siz', 'onlar', 'şu', 'şöyle', 'şimdi', 'sonra', 'önce'])

def remove_stopwords(text):
    words = str(text).split()
    filtered_words = [word for word in words if word not in dur_kelimeleri]
    return ' '.join(filtered_words) 

df["Final_Review"] = df["Clean_Review"].apply(remove_stopwords)

print("Orijinal, Temizlenmiş ve Stopwords'ten Arındırılmış Metin Örnekleri:")
print(df[['Review', 'Clean_Review', 'Final_Review', 'Sentiment']].head(10))


df_temiz = df[['Final_Review', 'Sentiment']]
df_temiz.to_csv("temizlenmis_yorumlar.csv", index=False, encoding='utf-8')
print("Temizlenmiş veri 'temizlenmis_yorumlar.csv' adıyla proje klasörüne kaydedildi.")

#4. TF-IDF vektörleştirme ve model eğitimi için veri setini hazırlama.

X = df["Final_Review"].fillna("")
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
tfidf  = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Eğitim matrisi boyutu: {X_train_tfidf.shape}")
print(f"Test matrisi boyutu: {X_test_tfidf.shape}")

#5. Model eğitimi ve değerlendirme.

optimize_agirlik = 0.20

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=optimize_agirlik,
    random_state=42,
    eval_metric='logloss',
    n_estimators=100
)

print("Model eğitimi başlatılıyor...")
xgb_model.fit(X_train_tfidf, y_train)

y_pred = xgb_model.predict(X_test_tfidf)

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

print("\nKarmaşıklık Matrisi:")
print(confusion_matrix(y_test, y_pred))

# 6. LIME ile model açıklaması.

explainer = LimeTextExplainer(class_names=['Olumsuz', 'Olumlu'])

def predictor_opt(texts):
    vec = tfidf.transform(texts)
    return xgb_model.predict_proba(vec)

yanlis_tahmin_indexleri = X_test[(y_test ==0) & (y_pred == 1)].index
ornek_index = yanlis_tahmin_indexleri[0]
ornek_metni = X_test.loc[ornek_index]

gercek_etiket = y_test.loc[ornek_index]
sira_no = list(X_test.index).index(ornek_index)
tahmin_edilen = y_pred[sira_no]

print(f"\nÖrnek Metin: {ornek_metni}")
print(f"Gerçek Etiket: {'Olumsuz' if gercek_etiket == 0 else 'Olumlu'}")
print(f"Tahmin Edilen Etiket: {'Olumsuz' if tahmin_edilen == 0 else 'Olumlu'}")

exp = explainer.explain_instance(ornek_metni, predictor_opt, num_features=6)

fig = exp.as_pyplot_figure()
plt.title("LIME Açıklaması")
plt.tight_layout()
plt.show()

# 7. SHAP ile model açıklaması.

explainer_shap = shap.TreeExplainer(xgb_model)
ornek_vektor_2d = X_test_tfidf[sira_no].toarray()

shap_values_exp = explainer_shap(ornek_vektor_2d)

shap_values_exp.feature_names = list(tfidf.get_feature_names_out())

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_exp[0])
plt.tight_layout()
plt.show()