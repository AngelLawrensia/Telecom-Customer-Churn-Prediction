
#import yang dibutuhkan
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

#Ini buat nampilin judul tab browser
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

#Menyimpan state sementara sehingga dapat direset 
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

#Tombol Reset, jadi alurnya ketika tombol ditekan -> key uploader berubah 
#-> Streamlit anggap ada uploader baru, sehingga file lama reset -> app rerun
if st.button("Refresh / Reset App"):
    st.session_state.uploader_key += 1
    st.rerun()

#Melakukan load model dan lain lain. 
model = joblib.load("models/randomforest_final.pkl") #Model Random Forest
scaler = joblib.load("models/scaler.pkl") #Normalisasi Data
columns = joblib.load("models/columns.pkl") #Urutan Feature Training
explainer = shap.Explainer(model)

#Ini buat judul di halaman sama tulisan dibawahnya 
st.title("Customer Churn Prediction")
st.markdown("Prediksi customer churn menggunakan Random Forest dan SHAP Explainability")

#Untuk upload csv atau xlsx
st.write("Upload data customer dalam format CSV atau Excel")

uploaded_file = st.file_uploader(
    "Upload File",
    type=["csv", "xlsx"],
    key=st.session_state.uploader_key #biar reset diatas bisa bekerja makanya ada key
)

#Ini untuk template csv dengan nama kolom-kolomnya 
raw_template = [
    "CustomerID",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method"
]

#Membuat dataframe kosong
template_df = pd.DataFrame(columns=raw_template)

#Tombol download
st.download_button(
    "Download Template CSV",
    data=template_df.to_csv(index=False),
    file_name="template_customer_churn.csv",
    mime="text/csv"
)

#Kode di bawah baru jalan kalau user upload file
if uploaded_file is not None:

    #Lebih ke milah sih jadi kalau csv pakai yang fungsi csv dan sebaliknya
    if uploaded_file.name.endswith(".csv"):
        input_df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        input_df = pd.read_excel(uploaded_file)
        
    #fungsinya untuk proses kolom yang diupload
    input_df = input_df.dropna(axis=1, how="all") #hapus kolom yang kosong total
    input_df = input_df.loc[:, ~input_df.columns.duplicated()] #hapus kolom duplikat
    input_df = input_df.loc[:, input_df.columns.notna()] #hapus kolom nama null

    st.subheader("Uploaded Data") #Menampilkan tabel
    st.dataframe(input_df) 

    #Expander = dropdown lipat. Digunakan buat debugging: Menampilkan nama kolom untuk cek mismatch
    with st.expander("Debug Kolom"): 
        st.write(input_df.columns.tolist())

    #Tombol prediksi akan muncul apabila tabel sudah terupload
    if st.button("Prediksi"):

        row_count = len(input_df)

        st.write(f"Jumlah data: {row_count}")

        #Disini didatasetnya kan ada ID nah ini biar id ga baca ketika dilakukan prediksi, disimpan dulu
        if "CustomerID" in input_df.columns:

            customer_ids = input_df["CustomerID"]

            input_df = input_df.drop(columns=["CustomerID"])

        #Kalau misalkan ga ada jadi bakal generate ID sendiri buat kebutuhan
        #Biasa digunakan untuk antisipasi seperti user lupa input. Untuk fallback atau mekanisme cadangan
        else:

            customer_ids = pd.Series(
                range(1, len(input_df) + 1),
                name="CustomerID"
            )

        #Untuk melakukan enkoding karna model cuma bisa membaca angka
        input_encoded = pd.get_dummies(input_df)

        # Cari kolom training yang hilang
        missing_cols = [
            col for col in columns
            if col not in input_encoded.columns
        ]

        #Apabila ditemukan maka ditambahkan agar bisa sama dengan feature yang ditraining
        for col in missing_cols:
            input_encoded[col] = 0

        #cari feature tambahan
        extra_cols = [
            col for col in input_encoded.columns
            if col not in columns
        ]

        #Buang Feature Asing
        if len(extra_cols) > 0:
            input_encoded = input_encoded.drop(columns=extra_cols)

        #Model membaca posisi feature
        input_encoded = input_encoded[columns]

        #Debug Encoding 
        with st.expander("Debug Encoding"):
            #Menampilkan fitur hilang
            st.write("Missing columns:", missing_cols)
            #Menampilkan fitur asing
            st.write("Extra columns:", extra_cols)
            #menampilkan hasil data encoding
            st.write(input_encoded.head())

        #Normalisasi data
        #Menyamakan skala fitur dan mencegah fitur besar mendominasi
        input_scaled = scaler.transform(input_encoded)

        input_scaled_df = pd.DataFrame(
            input_scaled,
            columns=columns
        )

        #Single Prediction jadi apabila file yang diupload cuma satu baris
        #Maka si ini akan masuk mode single kemudian dia akan menampilkan output untuk satuan
        #Jadi hasilnya langsung tertampil disana
        if row_count == 1:

            st.info("Mode: Single Prediction")

            prediction = model.predict(input_scaled)[0]

            probability = model.predict_proba(input_scaled)[0][1]

            st.subheader("Hasil Prediksi")

            col1, col2 = st.columns(2)

            with col1:

                if prediction == 1:
                    st.error("⚠️ Customer Akan Churn")

                else:
                    st.success("✅ Customer Tidak Churn")

            with col2:

                st.metric(
                    "Probabilitas Churn",
                    f"{probability:.2%}"
                )

            #Untuk ini adalah ketika sudah upload dan sudah keluar prediksi br di proses shap dibawahnya
            #Untuk SHAP single akan menggunakan waterfall tujuannya untuk melihat fitur mana yang mendorong churn 
            #Dan fitur mana yang menurunkan churn tersebut.
            st.subheader("SHAP Explanation")

            plt.close("all")

            shap_values = explainer.shap_values(input_scaled_df)

            #FIX FORMAT RANDOM FOREST SHAP
            if isinstance(shap_values, list):
                    
                single_shap = shap_values[1][0]

                base_value = explainer.expected_value[1]

            else:

                single_shap = shap_values[0, :, 1]

                base_value = explainer.expected_value[1]

            fig = plt.figure()

            shap.waterfall_plot(
                shap.Explanation(
                    values=single_shap,
                    base_values=base_value,
                    data=input_scaled_df.iloc[0],
                    feature_names=columns
                ),
                show=False
            )

            st.pyplot(fig)

        #Batch Prediction
        #Ini terjadi ketika kolom yang diupload lebih dari 1, dan akan melakukan prediksi massal
        #Hasil prediksi akan ditampilkan dalam bentuk list churn dan juga persentasenya
        #Kedua kolom tersebut diselipkan pada samping CustomerID
        #Mempermudahkan user untuk lihat hasilnya tanpa tarik ke kanan
        else:

            st.info("Mode: Batch Prediction")

            predictions = model.predict(input_scaled)

            probabilities = model.predict_proba(input_scaled)[:, 1]

            result_df = input_df.copy()

            result_df.insert(0, "CustomerID", customer_ids)

            result_df["Prediction"] = predictions

            result_df["Probability"] = probabilities

            result_df["Label"] = result_df["Prediction"].map({
                0: "Tidak Churn",
                1: "Churn"
            })

            #Disini pengurutan kolomnya agar mudah terlihat
            cols = result_df.columns.tolist()

            new_order = [
                "CustomerID",
                "Label",
                "Probability"
            ] + [
                c for c in cols
                if c not in [
                    "CustomerID",
                    "Label",
                    "Probability"
                ]
            ]

            result_df = result_df[new_order]

            #Menampilkan tabelnya
            st.subheader("Hasil Batch Prediction")

            st.dataframe(result_df)

            #Hasil prediksi dapat didownload dalam bentuk csv
            st.download_button(
                "Download Hasil Prediksi",
                result_df.to_csv(index=False),
                "hasil_prediksi.csv",
                mime="text/csv",
                on_click="ignore"
            )

            st.markdown("---")

            #Setelah hasilnya keluar juga menampilkan disini secara konklusi
            #Jadi total yang churn dan engga nampak
            st.subheader("Summary")

            churn_count = (predictions == 1).sum()

            non_churn_count = (predictions == 0).sum()

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Churn", churn_count)

            with col2:
                st.metric("Tidak Churn", non_churn_count)

            #Untuk Batch Prediction menggunakan SHAP Global
            #Untuk melihat fitur yang paling berpengaruh secara keseluruhan
            st.subheader("SHAP Global Feature Importance")

            plt.close("all")

            shap_values = explainer.shap_values(input_scaled_df)

            # FIX FORMAT RANDOM FOREST SHAP
            if isinstance(shap_values, list):

                shap_plot_values = shap_values[1]

            else:

                shap_plot_values = shap_values[:, :, 1]

            fig = plt.figure()

            shap.summary_plot(
                shap_plot_values,
                input_scaled_df,
                show=False
            )

            st.pyplot(fig)

            #Menghitung rata-rata absolut SHAP value
            #Semakin besar nilainya maka semakin penting fitur tersebut terhadap prediksi
            importance = np.abs(
                shap_plot_values
            ).mean(axis=0)

            importance_df = pd.DataFrame({
                "Feature": columns,
                "Importance": importance
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            st.subheader("Feature Importance")

            st.dataframe(importance_df)
