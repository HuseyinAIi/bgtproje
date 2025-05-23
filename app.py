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
