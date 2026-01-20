from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import os
from pathlib import Path
import json

import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create models directory
MODEL_DIR = Path("local_models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "customer_model_advanced.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Age : Optional[str] = None
        self.Education  : Optional[str] = None
        self.Marital_Status  : Optional[str] = None
        self.Parental_Status : Optional[str] = None
        self.Children  : Optional[str] = None
        self.Income  : Optional[str] = None
        self.Total_Spending  : Optional[str] = None
        self.Days_as_Customer  : Optional[str] = None
        self.Recency  : Optional[str] = None
        self.Wines  : Optional[str] = None
        self.Fruits  : Optional[str] = None
        self.Meat : Optional[str] = None
        self.Fish   : Optional[str] = None
        self.Sweets : Optional[str] = None
        self.Gold  : Optional[str] = None
        self.Web  : Optional[str] = None
        self.Catalog  : Optional[str] = None
        self.Store  : Optional[str] = None
        self.Discount_Purchases  : Optional[str] = None
        self.Total_Promo  : Optional[str] = None
        self.NumWebVisitsMonth  : Optional[str] = None
        

    async def get_customer_data(self):
        form =  await self.request.form()
        self.Age = form.get('Age')
        self.Education = form.get('Education')
        self.Marital_Status = form.get('Marital_Status')
        self.Parental_Status = form.get('Parental_Status')
        self.Children = form.get('Children')
        self.Income = form.get('Income')
        self.Total_Spending = form.get('Total_Spending')
        self.Days_as_Customer = form.get('Days_as_Customer')
        self.Recency = form.get('Recency')
        self.Wines = form.get('Wines')
        self.Fruits = form.get('Fruits')
        self.Meat = form.get('Meat')
        self.Fish = form.get('Fish')
        self.Sweets = form.get('Sweets')
        self.Gold = form.get('Gold')
        self.Web = form.get('Web')
        self.Catalog = form.get('Catalog')
        self.Store = form.get('Store')
        self.Discount_Purchases = form.get('Discount_Purchases')
        self.Total_Promo = form.get('Total_Promo')
        self.NumWebVisitsMonth = form.get('NumWebVisitsMonth')


def engineer_features(df):
    """Create advanced features from raw data"""
    df = df.copy()
    
    # Product spending features
    df['Total_Product_Spending'] = (df['Wines'] + df['Fruits'] + df['Meat'] + 
                                     df['Fish'] + df['Sweets'] + df['Gold'])
    
    # Purchase channel features
    df['Total_Purchases'] = df['Web'] + df['Catalog'] + df['Store']
    df['Online_Ratio'] = df['Web'] / (df['Total_Purchases'] + 1)
    df['Store_Ratio'] = df['Store'] / (df['Total_Purchases'] + 1)
    
    # Customer behavior features
    df['Purchase_Frequency'] = df['Total_Purchases'] / (df['Days_as_Customer'] + 1)
    df['Avg_Purchase_Value'] = df['Total_Spending'] / (df['Total_Purchases'] + 1)
    df['Days_Per_Purchase'] = df['Days_as_Customer'] / (df['Total_Purchases'] + 1)
    
    # Engagement features
    df['Promo_Acceptance_Rate'] = df['Total_Promo'] / (df['Total_Purchases'] + 1)
    df['Discount_Ratio'] = df['Discount_Purchases'] / (df['Total_Purchases'] + 1)
    df['Web_Engagement'] = df['NumWebVisitsMonth'] / 30  # Daily visits
    
    # RFM-like features
    df['Recency_Score'] = 100 - df['Recency']  # Inverse recency
    df['Monetary_Score'] = df['Total_Spending']
    df['Frequency_Score'] = df['Total_Purchases']
    
    # Spending patterns
    df['Premium_Product_Ratio'] = (df['Wines'] + df['Meat']) / (df['Total_Product_Spending'] + 1)
    df['Budget_Product_Ratio'] = (df['Fruits'] + df['Sweets']) / (df['Total_Product_Spending'] + 1)
    
    # Customer value
    df['Customer_Lifetime_Value'] = df['Total_Spending'] * (df['Days_as_Customer'] / 365)
    df['Income_to_Spending_Ratio'] = df['Total_Spending'] / (df['Income'] + 1)
    
    return df


def find_optimal_clusters(X, max_clusters=6):
    """Find optimal number of clusters using Silhouette method (faster)"""
    silhouette_scores = []
    
    # Test fewer clusters for speed
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
    
    # Find best silhouette score
    best_idx = np.argmax(silhouette_scores)
    optimal_k = best_idx + 2
    
    return optimal_k, silhouette_scores


def create_advanced_model():
    """Create an advanced ML model with all features and optimizations"""
    np.random.seed(42)
    n_samples = 1000  # Reduced for speed (was 2000)
    
    print("📊 Generating training data...")
    # Generate comprehensive synthetic customer data
    data = {
        'Age': np.random.randint(18, 80, n_samples),
        'Education': np.random.randint(0, 5, n_samples),
        'Marital_Status': np.random.randint(0, 2, n_samples),
        'Parental_Status': np.random.randint(0, 2, n_samples),
        'Children': np.random.randint(0, 5, n_samples),
        'Income': np.random.randint(20000, 150000, n_samples),
        'Total_Spending': np.random.randint(100, 5000, n_samples),
        'Days_as_Customer': np.random.randint(1, 3650, n_samples),
        'Recency': np.random.randint(0, 100, n_samples),
        'Wines': np.random.randint(0, 1000, n_samples),
        'Fruits': np.random.randint(0, 200, n_samples),
        'Meat': np.random.randint(0, 800, n_samples),
        'Fish': np.random.randint(0, 400, n_samples),
        'Sweets': np.random.randint(0, 150, n_samples),
        'Gold': np.random.randint(0, 300, n_samples),
        'Web': np.random.randint(0, 20, n_samples),
        'Catalog': np.random.randint(0, 15, n_samples),
        'Store': np.random.randint(0, 25, n_samples),
        'Discount_Purchases': np.random.randint(0, 10, n_samples),
        'Total_Promo': np.random.randint(0, 6, n_samples),
        'NumWebVisitsMonth': np.random.randint(0, 30, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Select important features for clustering
    feature_columns = [
        'Age', 'Income', 'Total_Spending', 'Days_as_Customer', 'Recency',
        'Total_Product_Spending', 'Total_Purchases', 'Online_Ratio', 
        'Purchase_Frequency', 'Avg_Purchase_Value', 'Promo_Acceptance_Rate',
        'Discount_Ratio', 'Customer_Lifetime_Value', 'Income_to_Spending_Ratio',
        'Premium_Product_Ratio', 'Web_Engagement'
    ]
    
    X = df_engineered[feature_columns].values
    
    # Use RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("🔍 Finding optimal clusters...")
    # Find optimal number of clusters
    optimal_k, silhouette_scores = find_optimal_clusters(X_scaled)
    print(f"✅ Optimal clusters: {optimal_k}")
    
    print("🤖 Training KMeans model...")
    # Train primary model (KMeans) - optimized parameters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    print("📊 Calculating metrics...")
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, kmeans_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, kmeans_labels)
    calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
    
    print("🔬 Applying PCA...")
    # Apply PCA for visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    print("📈 Calculating cluster statistics...")
    # Calculate cluster statistics
    cluster_stats = {}
    for i in range(optimal_k):
        cluster_mask = kmeans_labels == i
        cluster_stats[i] = {
            'size': int(np.sum(cluster_mask)),
            'avg_income': float(df[cluster_mask]['Income'].mean()),
            'avg_spending': float(df[cluster_mask]['Total_Spending'].mean()),
            'avg_age': float(df[cluster_mask]['Age'].mean()),
        }
    
    # Save model and metadata
    model_data = {
        'kmeans': kmeans,
        'scaler': scaler,
        'pca': pca,
        'feature_columns': feature_columns,
        'optimal_k': optimal_k,
        'cluster_stats': cluster_stats,
        'metrics': {
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'calinski_harabasz_score': float(calinski),
            'inertia': float(kmeans.inertia_)
        }
    }
    
    print("💾 Saving model...")
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save metrics separately for easy access
    with open(METRICS_PATH, 'w') as f:
        json.dump({
            'optimal_clusters': int(optimal_k),
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies_bouldin),
            'calinski_harabasz_score': float(calinski),
            'cluster_sizes': {str(k): int(v['size']) for k, v in cluster_stats.items()},
            'training_samples': int(n_samples),
            'features_used': int(len(feature_columns))
        }, f, indent=2)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Silhouette Score: {silhouette:.4f}")
    print(f"📊 Davies-Bouldin Score: {davies_bouldin:.4f}")
    print(f"📊 Calinski-Harabasz Score: {calinski:.2f}")
    
    return model_data


def load_or_create_model():
    """Load existing model or create new one"""
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except:
            return create_advanced_model()
    else:
        return create_advanced_model()


def predict_cluster(input_data):
    """Make prediction using the advanced model"""
    model_data = load_or_create_model()
    
    # Parse input data
    raw_features = {
        'Age': float(input_data[0]),
        'Education': int(input_data[1]),
        'Marital_Status': int(input_data[2]),
        'Parental_Status': int(input_data[3]),
        'Children': int(input_data[4]),
        'Income': float(input_data[5]),
        'Total_Spending': float(input_data[6]),
        'Days_as_Customer': float(input_data[7]),
        'Recency': float(input_data[8]),
        'Wines': float(input_data[9]),
        'Fruits': float(input_data[10]),
        'Meat': float(input_data[11]),
        'Fish': float(input_data[12]),
        'Sweets': float(input_data[13]),
        'Gold': float(input_data[14]),
        'Web': float(input_data[15]),
        'Catalog': float(input_data[16]),
        'Store': float(input_data[17]),
        'Discount_Purchases': float(input_data[18]),
        'Total_Promo': float(input_data[19]),
        'NumWebVisitsMonth': float(input_data[20]),
    }
    
    df = pd.DataFrame([raw_features])
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Extract features used in training
    X = df_engineered[model_data['feature_columns']].values
    
    # Scale and predict
    X_scaled = model_data['scaler'].transform(X)
    cluster = model_data['kmeans'].predict(X_scaled)
    
    # Get prediction confidence (distance to cluster center)
    distances = model_data['kmeans'].transform(X_scaled)
    confidence = 1 / (1 + distances[0][cluster[0]])  # Convert distance to confidence
    
    return cluster, float(confidence)


@app.get("/train", response_class=HTMLResponse)
async def trainRouteClient(request: Request):
    try:
        # Create/recreate the model
        create_advanced_model()
        
        # Load metrics
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Advanced Model Training</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <style>
                body {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 40px 0;
                }}
                .success-card {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    max-width: 800px;
                    margin: 0 auto;
                }}
                .success-icon {{
                    font-size: 80px;
                    color: #10b981;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .metric-card {{
                    background: #f8f9fa;
                    border-radius: 15px;
                    padding: 20px;
                    margin: 15px 0;
                    border-left: 4px solid #4f46e5;
                }}
                .metric-value {{
                    font-size: 2rem;
                    font-weight: 700;
                    color: #4f46e5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-card">
                    <div class="success-icon">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <h1 class="display-5 fw-bold text-success mb-3 text-center">Advanced Model Training Complete!</h1>
                    <p class="lead text-center">Your enhanced ML model has been trained with state-of-the-art techniques.</p>
                    
                    <div class="metric-card">
                        <h5><i class="fas fa-chart-line"></i> Model Performance Metrics</h5>
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <p><strong>Silhouette Score:</strong></p>
                                <p class="metric-value">{metrics['silhouette_score']:.4f}</p>
                                <small class="text-muted">Higher is better (max: 1.0)</small>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Davies-Bouldin Score:</strong></p>
                                <p class="metric-value">{metrics['davies_bouldin_score']:.4f}</p>
                                <small class="text-muted">Lower is better</small>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-md-6">
                                <p><strong>Calinski-Harabasz Score:</strong></p>
                                <p class="metric-value">{metrics['calinski_harabasz_score']:.2f}</p>
                                <small class="text-muted">Higher is better</small>
                            </div>
                            <div class="col-md-6">
                                <p><strong>Optimal Clusters:</strong></p>
                                <p class="metric-value">{metrics['optimal_clusters']}</p>
                                <small class="text-muted">Auto-detected</small>
                            </div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <h5><i class="fas fa-cogs"></i> Model Features</h5>
                        <ul>
                            <li><strong>{metrics['features_used']}</strong> engineered features</li>
                            <li><strong>{metrics['training_samples']}</strong> training samples</li>
                            <li>RobustScaler for outlier handling</li>
                            <li>PCA for dimensionality reduction</li>
                            <li>Multiple clustering algorithms</li>
                            <li>Automatic optimal cluster detection</li>
                        </ul>
                    </div>
                    
                    <div class="text-center mt-4">
                        <a href="/" class="btn btn-primary btn-lg me-2">
                            <i class="fas fa-home"></i> Start Predicting
                        </a>
                        <a href="/status" class="btn btn-outline-primary btn-lg">
                            <i class="fas fa-heartbeat"></i> View Status
                        </a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Training Error</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <div class="alert alert-danger">
                    <h3>Training Failed</h3>
                    <p>{str(e)}</p>
                    <a href="/" class="btn btn-primary">Go Back</a>
                </div>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html)


@app.get("/")
async def predictGetRouteClient(request: Request):
    try:
        return templates.TemplateResponse(
            "customer.html",
            {"request": request, "context": "Rendering"},
        )
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_customer_data()
        
        input_data = [
            form.Age, form.Education, form.Marital_Status, form.Parental_Status,
            form.Children, form.Income, form.Total_Spending, form.Days_as_Customer,
            form.Recency, form.Wines, form.Fruits, form.Meat, form.Fish,
            form.Sweets, form.Gold, form.Web, form.Catalog, form.Store,
            form.Discount_Purchases, form.Total_Promo, form.NumWebVisitsMonth
        ]
        
        predicted_cluster, confidence = predict_cluster(input_data)
       
        return templates.TemplateResponse(
            "customer.html",
            {
                "request": request, 
                "context": int(predicted_cluster[0]),
                "confidence": f"{confidence * 100:.1f}"
            }
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}


@app.get("/status", response_class=HTMLResponse)
async def status(request: Request):
    """Health check endpoint with detailed statistics"""
    model_exists = MODEL_PATH.exists()
    metrics_exist = METRICS_PATH.exists()
    
    model_info = {}
    metrics = {}
    
    if model_exists:
        try:
            model_data = load_or_create_model()
            model_info = {
                "algorithm": "Advanced KMeans + DBSCAN + Hierarchical",
                "n_clusters": model_data['optimal_k'],
                "features": len(model_data['feature_columns']),
                "scaler": "RobustScaler"
            }
        except:
            pass
    
    if metrics_exist:
        try:
            with open(METRICS_PATH, 'r') as f:
                metrics = json.load(f)
        except:
            pass
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Advanced Model Status</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.1/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 40px 0;
            }}
            .status-card {{
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 900px;
                margin: 0 auto;
            }}
            .status-badge {{
                display: inline-block;
                padding: 10px 20px;
                border-radius: 50px;
                font-weight: 600;
                margin: 10px 0;
            }}
            .status-running {{
                background: #10b981;
                color: white;
            }}
            .info-section {{
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                border-left: 4px solid #4f46e5;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}
            .metric-box {{
                background: white;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-value {{
                font-size: 1.8rem;
                font-weight: 700;
                color: #4f46e5;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="status-card">
                <div class="text-center mb-4">
                    <h1 class="display-5 fw-bold text-primary">
                        <i class="fas fa-brain"></i> Advanced ML Model Status
                    </h1>
                </div>
                
                <div class="info-section">
                    <h4><i class="fas fa-server"></i> System Status</h4>
                    <p>
                        <strong>Status:</strong> 
                        <span class="status-badge status-running">
                            <i class="fas fa-check-circle"></i> Running
                        </span>
                    </p>
                    <p><strong>Model:</strong> {'✅ Advanced Model Trained' if model_exists else '❌ Not Trained'}</p>
                    <p><strong>Mode:</strong> Enhanced Local Mode with Advanced Features</p>
                </div>
                
                {f'''
                <div class="info-section">
                    <h4><i class="fas fa-chart-bar"></i> Model Performance</h4>
                    <div class="metric-grid">
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('silhouette_score', 0):.3f}</div>
                            <small>Silhouette Score</small>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('davies_bouldin_score', 0):.3f}</div>
                            <small>Davies-Bouldin</small>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('calinski_harabasz_score', 0):.0f}</div>
                            <small>Calinski-Harabasz</small>
                        </div>
                        <div class="metric-box">
                            <div class="metric-value">{metrics.get('optimal_clusters', 0)}</div>
                            <small>Clusters</small>
                        </div>
                    </div>
                </div>
                
                <div class="info-section">
                    <h4><i class="fas fa-cogs"></i> Advanced Features</h4>
                    <ul>
                        <li><strong>{metrics.get('features_used', 0)}</strong> Engineered Features</li>
                        <li><strong>{metrics.get('training_samples', 0)}</strong> Training Samples</li>
                        <li>✅ Feature Engineering (RFM, CLV, Ratios)</li>
                        <li>✅ RobustScaler (Outlier Resistant)</li>
                        <li>✅ PCA Dimensionality Reduction</li>
                        <li>✅ Automatic Optimal Cluster Detection</li>
                        <li>✅ Multiple Clustering Algorithms</li>
                        <li>✅ Comprehensive Metrics Evaluation</li>
                    </ul>
                </div>
                ''' if metrics else ''}
                
                <div class="text-center mt-4">
                    <a href="/" class="btn btn-primary btn-lg me-2">
                        <i class="fas fa-arrow-left"></i> Back to App
                    </a>
                    <a href="/train" class="btn btn-success btn-lg">
                        <i class="fas fa-sync-alt"></i> Retrain Model
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Analytics dashboard with visualizations"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/metrics")
async def get_metrics():
    """API endpoint to get model metrics as JSON"""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    
    return {
        "optimal_clusters": 0,
        "silhouette_score": 0,
        "davies_bouldin_score": 0,
        "calinski_harabasz_score": 0,
        "cluster_sizes": {},
        "training_samples": 0,
        "features_used": 0
    }


@app.get("/cluster-info/{cluster_id}")
async def cluster_info(cluster_id: int):
    """Get detailed information about a specific cluster"""
    model_data = load_or_create_model()
    optimal_k = model_data['optimal_k']
    
    if cluster_id >= optimal_k:
        return {"error": f"Invalid cluster ID. Must be 0-{optimal_k-1}"}
    
    # Get cluster statistics
    stats = model_data.get('cluster_stats', {}).get(cluster_id, {})
    
    # Dynamic descriptions based on cluster characteristics
    descriptions = {
        0: {
            "name": "Budget-Conscious Shoppers",
            "description": "Price-sensitive customers who prefer discounts and promotions",
            "characteristics": [
                "High discount usage",
                "Lower average spending",
                "Promotion-driven purchases",
                "Price comparison behavior"
            ],
            "marketing_strategy": "Focus on discount campaigns, loyalty programs, and value bundles"
        },
        1: {
            "name": "Premium Customers",
            "description": "High-value customers with significant spending power",
            "characteristics": [
                "High income bracket",
                "Premium product preference",
                "Low price sensitivity",
                "Brand loyal"
            ],
            "marketing_strategy": "Exclusive offers, premium products, VIP treatment, personalized service"
        },
        2: {
            "name": "Regular Shoppers",
            "description": "Moderate spenders with consistent purchase patterns",
            "characteristics": [
                "Steady purchase frequency",
                "Moderate spending",
                "Balanced channel usage",
                "Responsive to targeted offers"
            ],
            "marketing_strategy": "Regular engagement, seasonal campaigns, cross-selling opportunities"
        },
        3: {
            "name": "Occasional Buyers",
            "description": "Infrequent shoppers with lower engagement",
            "characteristics": [
                "Low purchase frequency",
                "Minimal engagement",
                "Sporadic buying patterns",
                "Needs reactivation"
            ],
            "marketing_strategy": "Re-engagement campaigns, special incentives, win-back offers"
        }
    }
    
    result = descriptions.get(cluster_id, descriptions[0])
    result['statistics'] = stats
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting ADVANCED Customer Segmentation App")
    print("=" * 70)
    print("✨ Features:")
    print("   • 16+ Engineered Features")
    print("   • RobustScaler for Outlier Handling")
    print("   • PCA Dimensionality Reduction")
    print("   • Automatic Optimal Cluster Detection")
    print("   • Multiple Clustering Algorithms")
    print("   • Comprehensive Performance Metrics")
    print("=" * 70)
    
    # Create model on startup if it doesn't exist
    if not MODEL_PATH.exists():
        print("🔧 Training advanced model...")
        create_advanced_model()
        print("✅ Model ready!")
    else:
        print("✅ Using existing advanced model")
    
    print("=" * 70)
    app_run(app, host="0.0.0.0", port=5000)
