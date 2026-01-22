from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from uvicorn import run as app_run
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# from src.pipeline.prediction_pipeline import PredictionPipeline # Removed to avoid AWS dependency
# from src.pipeline.train_pipeline import TrainPipeline # Removed to avoid AWS dependency
# from src.constant.application import * # Removed to avoid src dependency
from app_local import predict_cluster, create_advanced_model, DataForm, load_or_create_model, METRICS_PATH # Import local logic
import json

APP_HOST = "0.0.0.0"
APP_PORT = 5000

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

@app.get("/train")
async def trainRouteClient():
    try:
        # train_pipeline = TrainPipeline()
        # train_pipeline.run_pipeline()
        
        create_advanced_model() # Use local training logic instead

        return Response("Training successful (Local Advanced Model) !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


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
        
        input_data = [form.Age, 
                    form.Education, 
                    form.Marital_Status, 
                    form.Parental_Status, 
                    form.Children, 
                    form.Income, 
                    form.Total_Spending, 
                    form.Days_as_Customer, 
                    form.Recency, 
                    form.Wines, 
                    form.Fruits, 
                    form.Meat, 
                    form.Fish, 
                    form.Sweets, 
                    form.Gold, 
                    form.Web, 
                    form.Catalog, 
                    form.Store, 
                    form.Discount_Purchases, 
                    form.Total_Promo, 
                    form.NumWebVisitsMonth]
        
        # prediction_pipeline = PredictionPipeline()
        # predicted_cluster = prediction_pipeline.run_pipeline(input_data=input_data)
        
        # Use local prediction logic
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
         print(f"❌ Error during prediction: {e}")
         return templates.TemplateResponse(
            "customer.html",
            {
                "request": request, 
                "context": "Rendering",
                "error": str(e)
            }
        )


@app.get("/dashboard")
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
    app_run(app, host = APP_HOST, port =APP_PORT)
    
