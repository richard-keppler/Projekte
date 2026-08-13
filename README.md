# Machine Learning, Deep Learning & NLP – Projekte

Dieses Repository bündelt mehrere Projekte aus den Bereichen **Machine Learning, Deep Learning und Natural Language Processing**, die im Rahmen meines M.Sc.-Studiums *Angewandte Künstliche Intelligenz* entstanden sind. Schwerpunkte sind medizinische Bild- und Textklassifikation sowie Zeitreihen-Prognose.

**Verwendete Technologien:** Python · TensorFlow / Keras · PyTorch · Hugging Face Transformers · scikit-learn · XGBoost · Optuna · Pandas / NumPy · Matplotlib / Seaborn

---

## Projektübersicht

### 1. Pneumonia Detection
Binäre Klassifikation von Röntgen-Thoraxaufnahmen (*Normal* vs. *Pneumonie*).

- **Ansatz:** Selbst entwickeltes CNN mit vier Convolutional-Blöcken, Dropout und Early Stopping.
- **Datenpipeline:** Normalisierung der Pixelwerte sowie Datenaugmentierung (Flip, Rotation).
- **Technologien:** TensorFlow / Keras, NumPy, Matplotlib.
- 📄 [`ML Pneumonia Detection.py`](ML%20Pneumonia%20Detection.py)

### 2. Klassifikation von Lungenerkrankungen
Mehrklassen-Klassifikation von Röntgenbildern in **vier Krankheitsbilder**.

- **Ansatz:** CNN-LSTM-Hybridmodell mit systematischem Hyperparameter-Tuning.
- **Umgang mit unausgewogenen Klassen:** Class Weights und Datenaugmentierung (Albumentations).
- **Ergebnis:** rund **89 % Test-Genauigkeit**.
- **Technologien:** TensorFlow / Keras, Albumentations, OpenCV, NumPy, SHAP.
- 📓 [`ML Hyperparameter Tuning.ipynb`](ML%20Hyperparameter%20Tuning.ipynb)

### 3. Kategorisierung von Krankheiten
NLP-Pipeline zur automatischen Kategorisierung unstrukturierter medizinischer Texte in **fünf Krankheitskategorien**.

- **Ansatz:** Domänenspezifisches Sprachmodell (**Bio_ClinicalBERT**) für Fine-Tuning und Feature-Extraktion, kombiniert mit einem Stacking-Ensemble (XGBoost / Logistic Regression).
- **Umgang mit unausgewogenen Klassen:** SMOTE und Class Weights; Hyperparameter-Optimierung mit Optuna, Modellinterpretation mit SHAP.
- **Technologien:** PyTorch, Hugging Face Transformers, scikit-learn, XGBoost, Optuna, SHAP.
- 📓 [`NLP Kategorisierung Krankheit.ipynb`](NLP%20Kategorisierung%20Krankheit.ipynb)

### 4. Strompreis-Vorhersage
Prognose stündlicher Strompreise auf Basis historischer Energie- und Wetterdaten.

- **Datenaufbereitung:** Zusammenführung von Energie- und Wetter-Zeitreihen, ARIMA-basierte Imputation fehlender Werte, Sequenzbildung und MinMax-Skalierung.
- **Ansatz:** Vergleich zweier Deep-Learning-Architekturen (**LSTM vs. GRU**) mit Early Stopping.
- **Ergebnis:** Das LSTM lieferte die genaueren Prognosen (R² ≈ 0,44, MAE ≈ 5,2) gegenüber dem GRU (R² ≈ 0,16, MAE ≈ 6,5).
- **Technologien:** PyTorch, statsmodels (ARIMA), scikit-learn, Pandas, Matplotlib.
- 📓 [`DL Vorhersage von Strompreisen.ipynb`](DL%20Vorhersage%20von%20Strompreisen.ipynb)

---

*Die Projekte dienen der praktischen Vertiefung im Studium und sind bewusst als eigenständige Fallstudien angelegt.*
