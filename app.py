import streamlit as st
import pandas as pd
import plotly.express as px # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.preprocessing import LabelEncoder # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score # pyright: ignore[reportMissingModuleSource]

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Performance Predictor App")

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload student_mat.csv", type=["csv"])

def encode_data(df):
    le_dict = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    preview_option = st.radio("Preview:", ["Show all rows", "Show first 10 rows"])
    if preview_option == "Show all rows":
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df.head(10), use_container_width=True)

    if 'G3' not in df.columns:
        st.error("The dataset must contain a column named 'G3'.")
        st.stop()
    else:
        df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        selected_num_col = st.selectbox("Select numeric column for histogram", df.select_dtypes(include='number').columns)
        fig = px.histogram(df, x=selected_num_col, color='pass', barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        x_axis = st.selectbox("X-axis", df.select_dtypes(include='number').columns, index=0)
        y_axis = st.selectbox("Y-axis", df.select_dtypes(include='number').columns, index=1)
        fig2 = px.scatter(df, x=x_axis, y=y_axis, color='pass')
        st.plotly_chart(fig2, use_container_width=True)

    X = df.drop(columns=['G1', 'G2', 'G3', 'pass'])
    y = df['pass']
    X, label_encoders = encode_data(X)

    st.sidebar.header("Model")
    model_choice = st.sidebar.radio("Choose model:", ["Random Forest", "Logistic Regression"])

    if model_choice == "Random Forest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 300, 100, step=10)
        max_depth = st.sidebar.slider("max_depth", 1, 20, 5)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader("Model Evaluation")
    st.text("Classification Report")
    st.code(classification_report(y_test, y_pred), language='text')

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Fail", "Pass"], columns=["Predicted Fail", "Predicted Pass"])
    st.plotly_chart(px.imshow(cm_df, text_auto=True, color_continuous_scale="viridis"))

    roc_auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    st.plotly_chart(px.area(roc_df, x='FPR', y='TPR'))

    st.subheader("Feature Importances")
    if model_choice == "Random Forest":
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.plotly_chart(px.bar(importances))

    st.subheader("Predict Single Student")
    user_input = {}
    for col in X.columns:
        if col in label_encoders:
            user_input[col] = st.selectbox(f"{col}:", options=list(label_encoders[col].classes_))
        else:
            user_input[col] = st.slider(f"{col}:", int(df[col].min()), int(df[col].max()), int(df[col].mean()))

    user_df = pd.DataFrame([user_input])
    for col in label_encoders:
        user_df[col] = label_encoders[col].transform(user_df[col])

    prediction = model.predict(user_df)[0]
    prediction_proba = model.predict_proba(user_df)[0][1]

    st.success(f"The student is predicted to: {'Pass' if prediction == 1 else 'Fail'} (Probability: {prediction_proba:.2f})")

    st.subheader("Batch Prediction")
    if st.checkbox("Run batch predictions on entire dataset"):
        pred_df = df.copy()
        pred_X = X.copy()
        pred_df['Predicted'] = model.predict(pred_X)
        pred_df['Pass Probability'] = model.predict_proba(pred_X)[:, 1]
        st.dataframe(pred_df[['Predicted', 'Pass Probability']])
        csv = pred_df.to_csv(index=False)
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

else:
    st.warning("Please upload a CSV file to continue.")