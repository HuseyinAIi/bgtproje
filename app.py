import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
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
    # GitHub raw URL'si üzerinden veri çekiliyor
    input_file_path = 'https://raw.githubusercontent.com/HuseyinAliYigit/bgtproje/refs/heads/main/KDDTrain%2B.txt'
    
    try:
        # Veriyi pandas ile oku
        data = pd.read_csv(input_file_path, header=None)
        
        # Kategorik sütunlar için map kullanımı
        protocol_mapping = {'tcp': 0, 'udp': 1, 'icmp': 2}
        # Servis değerleri datasette çok olduğundan, eksik kalmaması için tüm servis isimlerini topluca etiketlemek gerekir; 
        # basitçe örneklemek için burada sınırlı sayıda servis var, eksik servisler NaN oluyor.
        service_unique = data[2].unique()
        service_mapping = {k: v for v, k in enumerate(service_unique)}
        flag_unique = data[3].unique()
        flag_mapping = {k: v for v, k in enumerate(flag_unique)}
        class_unique = data[41].unique()
        class_mapping = {k: v for v, k in enumerate(class_unique)}

        # Kategorik olan sütunları sayısallaştır
        data[1] = data[1].map(protocol_mapping)
        data[2] = data[2].map(service_mapping)
        data[3] = data[3].map(flag_mapping)
        data[41] = data[41].map(class_mapping)
        
        # Eksik sayısallaştırılmış değerlerde NaN oluşursa temizle
        data.dropna(inplace=True)
        
        # Verinin yarısını rastgele seç
        data = data.sample(frac=0.5, random_state=42).reset_index(drop=True)
        
        return data, class_mapping
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None, None

# Model eğitimi ve değerlendirme fonksiyonu
def train_and_evaluate(data):
    X = data.drop(columns=[41])
    y = data[41]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB()
    }

    results = {}

    for model_name, model in models.items():
        st.write(f"{model_name} modeli eğitiliyor...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy * 100
        st.write(f"{model_name} modeli değerlendirildi: Doğruluk = {results[model_name]:.2f}%")
    
    return results

# Saldırı türleri analizi fonksiyonu
def analyze_attack_types(data, class_mapping):
    reverse_class_mapping = {v: k for k, v in class_mapping.items()}
    attack_types = data[41].unique()
    success_rates = []
    total = len(data)
    for attack in attack_types:
        name = reverse_class_mapping.get(attack, "Unknown")
        count = len(data[data[41] == attack])
        rate = (count / total) * 100
        success_rates.append((name, rate))
    return success_rates

# Uygulama arayüzü
def main():
    st.title("🛡️ Ağ Trafiği Analizi ile Saldırı Tespiti")

    selected = option_menu(
        menu_title=None,
        options=["Ana Sayfa", "Model Karşılaştırma", "Saldırı Analizi", "Proje Bilgileri"],
        icons=["house", "graph-up", "shield-exclamation", "info-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    data, class_mapping = load_data()
    if data is None:
        return
    
    if selected == "Ana Sayfa":
        st.header("Hoş Geldiniz")
        st.markdown("""
        Bu uygulama, NSL-KDD veri setini kullanarak ağ saldırılarını tespit etmek için farklı makine öğrenmesi algoritmalarının performansını karşılaştırır.
        
        **Özellikler:**
        - 2 farklı makine öğrenmesi algoritmasının performans karşılaştırması
        - Saldırı türlerine göre dağılım analizi
        - Etkileşimli grafikler ve görselleştirmeler
        
        Soldaki menüden istediğiniz bölüme geçiş yapabilirsiniz.
        """)
        
        st.subheader("Veri Seti Önizleme")
        st.dataframe(data.head())
        
        st.subheader("Temel İstatistikler")
        st.write(data.describe())
    
    elif selected == "Model Karşılaştırma":
        st.header("Model Karşılaştırması")
        st.markdown("Farklı makine öğrenmesi algoritmalarının doğruluk oranlarını karşılaştırın.")
        
        selected_model = st.selectbox(
            "Bir model seçin",
            ["Hepsini Göster", "Decision Tree", "Naive Bayes"]
        )
        
        if st.button("Modelleri Çalıştır"):
            with st.spinner("Modeller eğitiliyor ve değerlendiriliyor..."):
                results = train_and_evaluate(data)
            st.success("Model değerlendirmesi tamamlandı!")
            
            if selected_model == "Hepsini Göster":
                fig = px.bar(
                    x=list(results.keys()),
                    y=list(results.values()),
                    title='Algoritmaların Doğruluğu',
                    labels={'x': 'Model', 'y': 'Doğruluk (%)'},
                    color=list(results.keys()),
                    text=list(results.values()),
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy (%)'])
                st.dataframe(results_df.style.highlight_max(axis=0))
            else:
                for i in range(101):
                    st.progress(i)
                    time.sleep(0.01)
                st.metric(
                    label=f"{selected_model} Model Doğruluğu",
                    value=f"{results[selected_model]:.2f}%"
                )
    
    elif selected == "Saldırı Analizi":
        st.header("Saldırı Türleri Analizi")
        st.markdown("Veri setindeki saldırı türlerinin dağılımını ve başarı oranlarını görselleştirin.")
        
        attack_success = analyze_attack_types(data, class_mapping)
        names, rates = zip(*attack_success)
        
        fig = px.pie(
            names=names,
            values=rates,
            title='Saldırı Türlerine Göre Dağılım',
            hole=0.3,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = px.bar(
            x=names,
            y=rates,
            title='Saldırı Türlerine Göre Başarı Oranları',
            labels={'x': 'Saldırı Türü', 'y': 'Oran (%)'},
            color=names,
            text=rates,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)
        
        attack_success_df = pd.DataFrame({'Saldırı Türü': names, 'Oran (%)': rates})
        st.dataframe(attack_success_df.style.background_gradient(cmap='Blues'))
    
    elif selected == "Proje Bilgileri":
        st.header("Proje Bilgileri")
        st.markdown("""
        ### Projenin Adı / Project Name
        **Ağ Trafiği Analizi ile Saldırı Tespiti**
        
        ### Projenin Amacı / Purpose of Project
        Bu projenin amacı, ağ trafiği verilerini kullanarak çeşitli saldırı türlerini tespit etmek ve makine öğrenmesi yöntemleri ile bu saldırıları sınıflandırmaktır. 
        
        ### Projenin İçeriği / Content of Project
        Bu projede, ağ trafiği verilerinden saldırıları tespit etmek amacıyla makine öğrenmesi teknikleri kullanılacaktır. Veri seti olarak NSL-KDD kullanılmıştır.
        
        ### Proje Adımları
        1. Veri Toplama
        2. Veri Ön İşleme
        3. Makine Öğrenmesi Modellerini Eğitme (Karar Ağacı, Naive Bayes)
        4. Model Testi ve Değerlendirmesi
        5. Sonuçların Yorumlanması
        
        ### Kullanılan Teknolojiler
        - Python, Pandas, Scikit-learn
        - Streamlit
        - Plotly, Seaborn, Matplotlib
        """)

if __name__ == "__main__":
    main()
