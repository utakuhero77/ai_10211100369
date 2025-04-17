import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import base64
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import base64
import os
import sys
import io
# Set page cfrom sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.decomposition import PCA
st.set_page_config(
    page_title="ML & AI Explorer",
    page_icon="🧠",
    layout="wide"
)

# Main title
st.title("Machine Learning & AI Explorer")
st.markdown("Explore various machine learning and AI techniques with interactive interfaces")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a section",
    ["Home", "Regression", "Clustering", "Neural Networks", "Large Language Models"]
)

# Home section
if app_mode == "Home":
    st.header("Welcome to ML & AI Explorer")
    st.markdown("""
    This application allows you to explore different machine learning and AI techniques:
    
    - **Regression**: Predict continuous values based on input features
    - **Clustering**: Group similar data points together
    - **Neural Networks**: Build and train neural networks for various tasks
    - **Large Language Models**: Interact with pre-trained language models
    
    Select a section from the sidebar to get started.
    """)
    
    st.image("https://via.placeholder.com/800x400.png?text=ML+%26+AI+Explorer", caption="Machine Learning & AI Explorer")

# Regression section
elif app_mode == "Regression":
    st.header("Regression Analysis")
    st.markdown("""
    Regression analysis helps predict continuous values based on input features.
    Upload your dataset to get started.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset successfully loaded!")
            
            # Data preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
            # Data info
            st.subheader("Dataset Information")
            buffer = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isna().sum()
            })
            st.dataframe(buffer)
            
            # Data preprocessing options
            st.subheader("Data Preprocessing")
            
            # Select target variable
            target_col = st.selectbox("Select the target variable (dependent variable)", df.columns)
            
            # Select features
            feature_cols = st.multiselect(
                "Select the features (independent variables)",
                [col for col in df.columns if col != target_col],
                default=[col for col in df.columns if col != target_col][:3]  # Default to first 3 features
            )
            
            # Handle missing values
            handle_missing = st.checkbox("Handle missing values")
            if handle_missing:
                missing_strategy = st.radio(
                    "Choose a strategy for handling missing values",
                    ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with zero"]
                )
                
                if missing_strategy == "Drop rows with missing values":
                    df = df.dropna(subset=feature_cols + [target_col])
                elif missing_strategy == "Fill with mean":
                    for col in feature_cols:
                        if df[col].dtype in ['float64', 'int64']:
                            df[col] = df[col].fillna(df[col].mean())
                elif missing_strategy == "Fill with median":
                    for col in feature_cols:
                        if df[col].dtype in ['float64', 'int64']:
                            df[col] = df[col].fillna(df[col].median())
                elif missing_strategy == "Fill with zero":
                    for col in feature_cols:
                        df[col] = df[col].fillna(0)
            
            # Train model button
            if st.button("Train Regression Model"):
                if len(feature_cols) > 0:
                    # Prepare data
                    X = df[feature_cols]
                    y = df[target_col]
                    
                    # Split data
                    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    # Train model
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Calculate metrics
                    mae_train = mean_absolute_error(y_train, y_pred_train)
                    mae_test = mean_absolute_error(y_test, y_pred_test)
                    mse_train = mean_squared_error(y_train, y_pred_train)
                    mse_test = mean_squared_error(y_test, y_pred_test)
                    rmse_train = np.sqrt(mse_train)
                    rmse_test = np.sqrt(mse_test)
                    r2_train = r2_score(y_train, y_pred_train)
                    r2_test = r2_score(y_test, y_pred_test)
                    
                    # Display coefficients
                    st.subheader("Model Coefficients")
                    coef_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Coefficient': model.coef_
                    })
                    st.dataframe(coef_df)
                    st.write(f"Intercept: {model.intercept_:.4f}")
                    
                    # Display metrics
                    st.subheader("Model Performance Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Training Set Metrics:")
                        st.write(f"MAE: {mae_train:.4f}")
                        st.write(f"MSE: {mse_train:.4f}")
                        st.write(f"RMSE: {rmse_train:.4f}")
                        st.write(f"R² Score: {r2_train:.4f}")
                    
                    with col2:
                        st.write("Test Set Metrics:")
                        st.write(f"MAE: {mae_test:.4f}")
                        st.write(f"MSE: {mse_test:.4f}")
                        st.write(f"RMSE: {rmse_test:.4f}")
                        st.write(f"R² Score: {r2_test:.4f}")
                    
                    # Visualization - Predictions vs Actual
                    st.subheader("Predictions vs Actual Values")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_test, y_pred_test, alpha=0.5)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r-', lw=2)
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Predictions vs Actual Values")
                    st.pyplot(fig)
                    
                    # Visualization - Residuals
                    st.subheader("Residuals Plot")
                    residuals = y_test - y_pred_test
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(y_pred_test, residuals, alpha=0.5)
                    ax.hlines(y=0, xmin=y_pred_test.min(), xmax=y_pred_test.max(), colors='r', linestyles='--')
                    ax.set_xlabel("Predicted Values")
                    ax.set_ylabel("Residuals")
                    ax.set_title("Residuals vs Predicted Values")
                    st.pyplot(fig)
                    
                    # Distribution of residuals
                    st.subheader("Distribution of Residuals")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(residuals, kde=True, ax=ax)
                    ax.set_xlabel("Residuals")
                    ax.set_title("Distribution of Residuals")
                    st.pyplot(fig)
                    
                    # Custom prediction section
                    st.subheader("Make Custom Predictions")
                    st.write("Enter values for your features:")
                    
                    custom_inputs = {}
                    for feature in feature_cols:
                        # Check if the feature is numeric
                        if df[feature].dtype in ['float64', 'int64']:
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            mean_val = float(df[feature].mean())
                            
                            # Handle extremely large ranges
                            if max_val - min_val > 1000:
                                step = (max_val - min_val) / 100
                            else:
                                step = (max_val - min_val) / 20
                            
                            custom_inputs[feature] = st.slider(
                                f"{feature}",
                                min_val,
                                max_val,
                                mean_val,
                                step
                            )
                        else:
                            # For categorical features, offer a selectbox
                            unique_values = df[feature].unique().tolist()
                            custom_inputs[feature] = st.selectbox(f"{feature}", unique_values)
                    
                    # Make prediction
                    if st.button("Predict"):
                        # Prepare input data
                        input_data = pd.DataFrame([custom_inputs])
                        
                        # Make prediction
                        prediction = model.predict(input_data)[0]
                        
                        # Display prediction
                        st.success(f"Predicted {target_col}: {prediction:.4f}")
                else:
                    st.error("Please select at least one feature for training.")
        
        except Exception as e:
            st.error(f"Error loading or processing the dataset: {e}")

# Add these imports at the top of your file


# In your main code, replace the placeholder Clustering section with this:

elif app_mode == "Clustering":
    st.title("🔍 AI/ML Explorer: Clustering Analysis")
    st.markdown("""
    This module allows you to perform K-Means clustering on your dataset. 
    Upload your data, select features, and explore the clusters.
    """)
    
    # Initialize session state for clustering
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    
    with st.sidebar:
        st.header("📊 Clustering Configuration")
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=['csv', 'xlsx'], key='clustering_uploader')
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success("Data loaded successfully!")
                
                # Show basic info
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Select features for clustering
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                features = st.multiselect(
                    "Select features for clustering",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))] if numeric_cols else []
                )
                
                # Number of clusters
                n_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="Select the number of clusters to create"
                )
                
                # Normalization option
                normalize = st.checkbox(
                    "Normalize features",
                    value=True,
                    help="Standardize features to have mean=0 and variance=1"
                )
                
                if st.button("Perform Clustering"):
                    if not features:
                        st.error("Please select at least one feature for clustering.")
                    else:
                        # Prepare data
                        X = df[features].dropna()
                        
                        # Normalize if requested
                        if normalize:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                        else:
                            X_scaled = X.values
                        
                        # Perform K-Means clustering
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # Store results
                        clustered_df = X.copy()
                        clustered_df['Cluster'] = clusters
                        
                        # For visualization, reduce dimensions if needed
                        if len(features) > 2:
                            pca = PCA(n_components=2)
                            vis_data = pca.fit_transform(X_scaled)
                            vis_df = pd.DataFrame(vis_data, columns=['PC1', 'PC2'])
                            vis_df['Cluster'] = clusters
                            x_col, y_col = 'PC1', 'PC2'
                        else:
                            vis_df = clustered_df.copy()
                            x_col, y_col = features[0], features[1] if len(features) > 1 else features[0]
                        
                        # Store in session state
                        st.session_state.clustered_df = clustered_df
                        st.session_state.vis_df = vis_df
                        st.session_state.x_col = x_col
                        st.session_state.y_col = y_col
                        st.session_state.n_clusters = n_clusters
                        st.session_state.features = features
                        st.session_state.clustering_done = True
                        
                        st.success("Clustering completed successfully!")
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Main content area
    if st.session_state.get('clustering_done', False):
        st.header("📊 Clustering Results")
        
        # Show cluster sizes
        cluster_sizes = st.session_state.clustered_df['Cluster'].value_counts().sort_index()
        st.write("**Cluster Sizes:**")
        st.dataframe(cluster_sizes)
        
        # Visualization
        st.subheader("Cluster Visualization")
        
        # Create interactive plot
        fig = px.scatter(
            st.session_state.vis_df,
            x=st.session_state.x_col,
            y=st.session_state.y_col,
            color='Cluster',
            color_continuous_scale=px.colors.qualitative.Plotly,
            title=f"K-Means Clustering (k={st.session_state.n_clusters})"
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            xaxis_title=st.session_state.x_col,
            yaxis_title=st.session_state.y_col,
            legend_title="Cluster"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = st.session_state.clustered_df.groupby('Cluster')[st.session_state.features].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        
        # Download clustered data
        st.subheader("Download Results")
        csv = st.session_state.clustered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download clustered data as CSV",
            data=csv,
            file_name='clustered_data.csv',
            mime='text/csv'
        )
    
    else:
        st.info("Please upload a dataset and configure clustering parameters in the sidebar.")




elif app_mode == "Neural Networks":
    st.header("🧠 Neural Network Trainer (with PyTorch)")

    st.markdown("Upload a classification dataset to build and train a simple Feedforward Neural Network.")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())

        target_column = st.selectbox("Select the target column", df.columns)

        # Hyperparameter options
        epochs = st.slider("Epochs", 5, 100, 20)
        learning_rate = st.number_input("Learning Rate", value=0.001)
        batch_size = st.slider("Batch Size", 8, 128, 32)

        if st.button("Train Model"):
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from torch.utils.data import DataLoader, TensorDataset
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import matplotlib.pyplot as plt

            # Preprocess
            X = df.drop(columns=[target_column])
            y = df[target_column]

            for col in X.select_dtypes(include='object').columns:
                X[col] = LabelEncoder().fit_transform(X[col])

            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)

            input_dim = X_train.shape[1]
            num_classes = len(set(y))

            class FeedforwardNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 64)
                    self.fc2 = nn.Linear(64, 32)
                    self.fc3 = nn.Linear(32, num_classes)

                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = torch.relu(self.fc2(x))
                    return self.fc3(x)

            model = FeedforwardNN()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_losses, val_losses = [], []
            train_accuracies, val_accuracies = [], []

            progress = st.progress(0)

            for epoch in range(epochs):
                model.train()
                correct, total, train_loss = 0, 0, 0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                train_accuracy = correct / total
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)

                # Validation
                model.eval()
                correct, total, val_loss = 0, 0, 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, 1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                val_accuracy = correct / total
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

                progress.progress((epoch + 1) / epochs)
                st.write(f"Epoch {epoch+1}/{epochs}: Train Acc={train_accuracy:.4f}, Val Acc={val_accuracy:.4f}")

            # Plot
            st.subheader("📊 Training Progress")
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))

            axs[0].plot(train_losses, label="Train Loss")
            axs[0].plot(val_losses, label="Val Loss")
            axs[0].legend()
            axs[0].set_title("Loss over Epochs")

            axs[1].plot(train_accuracies, label="Train Accuracy")
            axs[1].plot(val_accuracies, label="Val Accuracy")
            axs[1].legend()
            axs[1].set_title("Accuracy over Epochs")

            st.pyplot(fig)

            # Save model + scaler
            torch.save(model.state_dict(), "model.pth")
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['input_dim'] = input_dim

    # Prediction
    st.subheader("🧪 Make Predictions")
    if "model" in st.session_state:
        test_file = st.file_uploader("Upload test CSV for prediction", type=["csv"], key="predict")

        if test_file:
            test_df = pd.read_csv(test_file)
            st.write("Test Data Preview:", test_df.head())

            for col in test_df.select_dtypes(include='object').columns:
                test_df[col] = LabelEncoder().fit_transform(test_df[col])

            test_df = st.session_state['scaler'].transform(test_df)
            test_tensor = torch.tensor(test_df, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                outputs = model(test_tensor)
                _, preds = torch.max(outputs, 1)
                st.write("Predictions:", preds.numpy())



    
elif app_mode == "Large Language Model (LLM)":
    load_dotenv()
    QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    HF_API_KEY = os.getenv("HF_API_KEY")

    # Initialize Qdrant client
    qdrant = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)

    # Load embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_text_from_pdf(pdf_path):
        text = ""
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def split_text(text, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_text(text)
    
    def retrieve_relevant_chunks(query):
        query_embedding = embedding_model.encode([query])[0]
        search_results = qdrant.search(
        collection_name="handbook",
        query_vector=query_embedding.tolist(),
        limit=5
    )
        return [hit.payload["text"] for hit in search_results]

# Function to generate answer using the Hugging Face model
    def generate_answer(query, retrieved_text):
        prompt = (
        "You are an AI assistant that answers questions based on provided document excerpts. "
        "Your goal is to extract the most relevant information and provide a concise, factual summary.\n\n"
        "### Document Excerpts:\n"
        f"{retrieved_text}\n\n"
        "### User Question:\n"
        f"{query}\n\n"
        "### Instructions:\n"
        "- Identify the key points that directly answer the user's question.\n"
        "- Provide a *clear and structured summary* in *2-3 sentences*.\n"
        "- Avoid unnecessary details or repeating the text verbatim.\n"
        "- If the excerpts do not contain enough information, state: 'The document does not provide a clear answer to this question.'\n\n"
        "### Answer:"
    )

        response = requests.post(
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={
            "inputs": prompt,
            "parameters": {
                "temperature": 0.3,  # Low temp for accuracy
                "max_new_tokens": 150  # Controls response length
            }
        }
    )
  
        generated_text = response.json()[0]["generated_text"]
    
    # Split by "### Answer:" to isolate the actual response
        answer = generated_text.split("### Answer:")[-1].strip()
    
        return answer


    st.header("Large Language Model Q&A")


    # Extract text from the PDF and process
    pdf_text = extract_text_from_pdf("handbook.pdf")
    text_chunks = split_text(pdf_text)

    # Create embeddings for the document chunks and upload to Qdrant
    chunk_embeddings = np.array(embedding_model.encode(text_chunks))
    qdrant.recreate_collection(
        collection_name="handbook",
        vectors_config=VectorParams(
            size=chunk_embeddings.shape[1],
            distance=Distance.COSINE
        ),
    )

    # Upload embeddings to Qdrant
    qdrant.upload_points(
        collection_name="handbook",
        points=[
            PointStruct(id=i, vector=chunk_embeddings[i].tolist(), payload={"text": text_chunks[i]})
            for i in range(len(text_chunks))
        ]
    )

    st.subheader("Ask a Question")
    query = st.text_input("Enter your question")
    if st.button("Ask") and query:
        relevant_chunks = retrieve_relevant_chunks(query)
        rag_answer = generate_answer(query, "\n".join(relevant_chunks))
        st.write("Answer:", rag_answer)


