import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, 
    recall_score, 
    precision_score, 
    confusion_matrix
)
import streamlit as st
from streamlit_option_menu import option_menu
import time
import plotly.express as px

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="Ağ Trafiği Analizi ile Saldırı Tespiti",
    page_icon="🛡️",
    layout="wide"
)

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    input_file_path = 'https://raw.githubusercontent.com/HuseyinAliYigit/bgtproje/refs/heads/main/KDDTrain%2B.txt'  # Güncellenmiş dosya yolu
    
    try:
        # Veriyi pandas ile oku
        data = pd.read_csv(input_file_path, header=None)
        
        # Kategorik sütunları sayısallaştırmak için LabelEncoder kullanacağız
        label_encoder = LabelEncoder()

        # Kategorik olan sütunları etiketle (protocol_type, service, flag ve class)
        data[1] = label_encoder.fit_transform(data[1])  # protocol_type
        data[2] = label_encoder.fit_transform(data[2])  # service
        data[3] = label_encoder.fit_transform(data[3])  # flag
        data[41] = label_encoder.fit_transform(data[41])  # class (etiket)
        
        # Verinin yarısını rastgele seç
        data = data.sample(frac=0.5, random_state=42)
        
        return data
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None

# Model eğitimi ve değerlendirme fonksiyonu
def train_and_evaluate(data):
    # Veriyi özellikler (X) ve etiketler (y) olarak ayır
    X = data.drop(columns=[41])  # Özellikler (class sütunu hariç)
    y = data[41]  # Etiketler (class sütunu)

    # Eğitim ve test verilerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Modelleri tanımla
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    # Sonuçları depolayacağımız bir sözlük
    results = {}

    for model_name, model in models.items():
        # Modeli eğit
        st.write(f"{model_name} modeli eğitiliyor...")
        model.fit(X_train, y_train)
        
        # Model ile tahmin yap
        y_pred = model.predict(X_test)
        
        # Performans metriklerini hesapla
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy
        sensitivity = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix'i hesapla
        cm = confusion_matrix(y_test, y_pred)
        
        if cm.size == 4:  # 2x2 confusion matrix
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
        else:
            # Eğer yeterli sınıf yoksa, varsayılan değerler kullan
            tn, fp, fn, tp = 0, 0, 0, 0
            specificity = 0
            negative_predictive_value = 0

        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0
        positive_likelihood_ratio = sensitivity / false_positive_rate if false_positive_rate > 0 else 0
        negative_likelihood_ratio = false_negative_rate / specificity if specificity > 0 else 0
        diagnostic_odds_ratio = (tp * tn) / (fp * fn) if (fp * fn) > 0 else 0

        # Sonuçları kaydet
        results[model_name] = {
            "Doğruluk": accuracy * 100,
            "Hata Oranı": error_rate * 100,
            "Duyarlılık": sensitivity * 100,
            "Kesinlik": precision * 100,
            "Belirleyicilik": specificity * 100,
            "Pozitif Öngörü": precision * 100,
            "Negatif Öngörü": negative_predictive_value * 100,
            "Yanlış Pozitif Oranı": false_positive_rate * 100,
            "Yanlış Negatif Oranı": false_negative_rate * 100,
            "Pozitif Olabilirlik Oranı": positive_likelihood_ratio,
            "Negatif Olabilirlik Oranı": negative_likelihood_ratio,
            "Tanısal Üstünlük Oranı": diagnostic_odds_ratio
        }
        
        st.write(f"{model_name} modeli değerlendirildi: {results[model_name]}")

    return results, X, y

# Saldırı türlerini analiz etme fonksiyonu
def analyze_attack_types(data):
    # Saldırı türleri (class sütunu)
    attack_types = data[41].unique()  
    success_rates = []

    # Saldırı türlerinin isimlerini ve karşılık gelen sayıları eşleştiren bir sözlük
    attack_type_mapping = {
        0: 'normal',
        1: 'neptune',
        2: 'warezclient',
        3: 'ipsweep',
        4: 'portsweep',
        5: 'teardrop',
        6: 'nmap',
        7: 'satan',
        8: 'smurf',
        9: 'pod',
        10: 'back',
        11: 'guess_passwd',
        12: 'ftp_write',
        13: 'multihop',
        14: 'rootkit',
        15: 'buffer_overflow',
        16: 'imap',
        17: 'warezmaster',
        18: 'phf',
        19: 'land',
        20: 'loadmodule',
        21: 'spy',
        22: 'perl'
    }

    for attack in attack_types:
        # Saldırı türünün ismini almak için mapping sözlüğünü kullan
        attack_name = attack_type_mapping.get(attack, 'Unknown')
        success_rate = len(data[data[41] == attack]) / len(data) * 100
        success_rates.append((attack_name, success_rate))
    
    return success_rates

# Uygulama arayüzü
def main():
    # Başlık
    st.title("🛡️ Ağ Trafiği Analizi ile Saldırı Tespiti")
    
    # Yatay menü
    selected = option_menu(
        menu_title=None,
        options=["Ana Sayfa", "Model Karşılaştırma", "Saldırı Analizi", "Proje Bilgileri"],
        icons=["house", "graph-up", "shield-exclamation", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )
    
    # Veriyi yükle
    data = load_data()
    
    if data is None:
        return
    
    if selected == "Ana Sayfa":
        st.header("Hoş Geldiniz")
        st.markdown("""
        Bu uygulama, NSL-KDD veri setini kullanarak ağ saldırılarını tespit etmek için farklı makine öğrenmesi algoritmalarının performansını karşılaştırır.
        
        **Özellikler:**
        - 3 farklı makine öğrenmesi algoritmasının performans karşılaştırması
        - Saldırı türlerine göre dağılım analizi
        - Etkileşimli grafikler ve görselleştirmeler
        
        Soldaki menüden istediğiniz bölüme geçiş yapabilirsiniz.
        """)
        
        # Veri önizleme
        st.subheader("Veri Seti Önizleme")
        st.dataframe(data.head())
        
        # Temel istatistikler
        st.subheader("Temel İstatistikler")
        st.write(data.describe())
    
    elif selected == "Model Karşılaştırma":
        st.header("Model Karşılaştırması")
        st.markdown("Farklı makine öğrenmesi algoritmalarının doğruluk oranlarını karşılaştırın.")
        
        if st.button("Modelleri Çalıştır"):
            with st.spinner("Modeller eğitiliyor ve değerlendiriliyor..."):
                # Model eğitimi ve değerlendirme
                results, X, y = train_and_evaluate(data)
                
                # Sonuçları göster
                st.success("Model değerlendirmesi tamamlandı!")
                
                # Sonuçları DataFrame'e çevir
                results_df = pd.DataFrame(results).T
                
                # Tabloda en yüksek doğruluğa sahip modeli vurgula
                results_df['Doğruluk'] = results_df['Doğruluk'].astype(float)
                best_model = results_df['Doğruluk'].idxmax()
                results_df['En İyi Model'] = results_df.index == best_model
                
                # Sonuç tablosunu göster
                st.subheader("Model Performans Sonuçları")
                st.dataframe(results_df.style.apply(lambda x: ['background: lightgreen' if x.name == best_model else '' for _ in x], axis=1))
                
                # Doğruluk oranlarını görselleştir
                fig = px.bar(
                    results_df,
                    x=results_df.index,
                    y='Doğruluk',
                    title='Algoritmaların Doğruluğu',
                    labels={'x': 'Model', 'y': 'Doğruluk (%)'},
                    color='Doğruluk',
                    color_continuous_scale=px.colors.sequential.Viridis,
                    text='Doğruluk'
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                # Diğer metrikleri görselleştir
                metrics_to_plot = ['Hata Oranı', 'Duyarlılık', 'Kesinlik']
                metrics_fig = px.bar(
                    results_df,
                    x=results_df.index,
                    y=metrics_to_plot,
                    title='Diğer Performans Metrikleri',
                    labels={'x': 'Model', 'y': 'Değer'},
                    barmode='group',
                    text_auto=True
                )
                st.plotly_chart(metrics_fig, use_container_width=True)
    
    elif selected == "Saldırı Analizi":
        st.header("Saldırı Türleri Analizi")
        st.markdown("Veri setindeki saldırı türlerinin dağılımını ve başarı oranlarını görselleştirin.")
        
        success_rates = analyze_attack_types(data)
        
        # Saldırı türleri grafiği
        attack_names, rates = zip(*success_rates)  # İsimleri ve oranları ayır
        fig = px.pie(
            names=attack_names,
            values=rates,
            title='Saldırı Türlerine Göre Dağılım',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Çubuk grafik
        fig2 = px.bar(
            x=attack_names,
            y=rates,
            title='Saldırı Türlerine Göre Başarı Oranları',
            labels={'x': 'Saldırı Türü', 'y': 'Oran (%)'},
            color=attack_names,
            text=rates,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Saldırı türleri tablosu
        attack_success_df = pd.DataFrame({'Saldırı Türü': attack_names, 'Oran (%)': rates})
        st.dataframe(attack_success_df.style.background_gradient(cmap='Blues'))
    
    elif selected == "Proje Bilgileri":
        st.header("Proje Bilgileri")
        
        # Proje bilgileri
        st.markdown("""
        ### Projenin Adı / Project Name
        **Ağ Trafiği Analizi ile Saldırı Tespiti**
        
        ### Projenin Amacı / Purpose of Project
        Bu projenin amacı, ağ trafiği verilerini kullanarak çeşitli saldırı türlerini tespit etmek ve makine öğrenmesi yöntemleri ile bu saldırıları sınıflandırmaktır. 
        Proje, NSL-KDD veri seti kullanılarak ağ saldırılarını sınıflandıracak bir makine öğrenmesi modeli geliştirmeyi hedeflemektedir.
        
        ### Projenin İçeriği / Content of Project
        Bu projede, ağ trafiği verilerinden saldırıları tespit etmek amacıyla makine öğrenmesi teknikleri kullanılacaktır. 
        Veri seti olarak, yaygın olarak kullanılan ve ağ güvenliği araştırmalarında sıklıkla kullanılan NSL-KDD veri seti kullanılacaktır. 
        Veri seti, ağ trafiği ile ilgili çeşitli özellikler içerir ve bu özellikler saldırıları sınıflandırmak için kullanılır.
        
        **Proje adımlarını şu şekilde planlıyoruz:**
        1. **Veri Toplama:** NSL-KDD veri seti indirilecektir.
        2. **Veri Ön İşleme:** Kategorik veriler sayısal verilere dönüştürülecek, eksik veriler tamamlanacak ve veriler ölçeklendirilecektir.
        3. **Makine Öğrenmesi Modeli:** Karar ağaçları, Naive Bayes ve K-En Yakın Komşu gibi sınıflandırıcı algoritmalar kullanılacak ve model eğitilecektir.
        4. **Model Testi ve Değerlendirmesi:** Eğitim ve test setleri oluşturularak modelin doğruluğu değerlendirilecektir.
        5. **Sonuçların Yorumlanması:** Sonuçlar analiz edilip, saldırı tespitinin doğruluğu ve başarısı değerlendirilecektir.
        
        ### Kullanılan Teknolojiler
        - Python
        - Pandas, NumPy (Veri işleme)
        - Scikit-learn (Makine öğrenmesi)
        - Streamlit (Kullanıcı arayüzü)
        - Plotly, Matplotlib, Seaborn (Veri görselleştirme)
        """)

# Uygulamayı çalıştır
if __name__ == "__main__":
    main()
