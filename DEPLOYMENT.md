# 🚀 Deployment Guide

## How to Host Your Application Online

This guide explains how to deploy your FastAPI application to **Render.com** (Free Tier).

### 📋 Prerequisites
- GitHub Repository (Created)
- Render.com Account (Sign up with GitHub)

---

### ☁️ Step-by-Step Deployment on Render

1. **Go to Render Dashboard**
   - Visit [https://dashboard.render.com](https://dashboard.render.com)
   - Click **"New +"** button -> Select **"Web Service"**

2. **Connect Repository**
   - You will see a list of your GitHub repositories.
   - Find `customer_personality_segmentation` and click **"Connect"**.

3. **Configure Service**
   Fill in the details exactly as below:

   | Setting | Value |
   |---------|-------|
   | **Name** | `customer-segmentation-app` (or your choice) |
   | **Region** | Select nearest to you (e.g., Singapore) |
   | **Branch** | `main` |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn app_local:app --host 0.0.0.0 --port $PORT` |
   | **Instance Type** | `Free` |

4. **Environment Variables** (Optional)
   If you want to modify settings without changing code, you can add:
   - Key: `PYTHON_VERSION`
   - Value: `3.12.0` (or `3.11.5` if 3.12 fails)

5. **Deploy**
   - Click **"Create Web Service"**.
   - Watch the logs. It will install dependencies and start the server.
   - Once it says `Application startup complete`, your app is live!

---

### 🌐 Accessing Your App

Render will provide a URL (e.g., `https://customer-segmentation.onrender.com`).
- **Home**: `https://.../`
- **Docs**: `https://.../docs`
- **Dashboard**: `https://.../dashboard`

---

### ⚠️ Important Limitations (Free Tier)

Since we are using **SQLite** and **Local Files**:

1. **Data Reset**: Every time you deploy or the server restarts (spins down due to inactivity), the `predictions.db` file will be reset to its initial state.
2. **Model Persistence**: If you retrain the model online (`/train`), the new model version is saved to the container. If the container restarts, it reverts to the GitHub version of the model.
3. **Spin Down**: Free apps "sleep" after 15 minutes of inactivity. The first request might take 30-50 seconds to load while it wakes up.

**For a demo/portfolio, these limitations are completely acceptable!**

---

### 🔍 Troubleshooting

- **Build Failed?** Check the logs. If it complains about Python version, try adding the `PYTHON_VERSION` environment variable.
- **App Crashes?** Ensure the **Start Command** is exactly: `uvicorn app_local:app --host 0.0.0.0 --port $PORT`
