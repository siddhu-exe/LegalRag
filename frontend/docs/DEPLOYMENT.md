# LegalRAG Frontend — Production Deployment Runbook

This guide details how to build, test, and deploy the **LegalRAG** React 19 + Vite frontend across major cloud providers (Vercel, Netlify, Cloudflare Pages, Docker, AWS S3/CloudFront).

---

## 1. Pre-Deployment Verification Checklist

Before deploying, run the pre-deployment verification steps locally:

- [ ] **Dependencies installed cleanly**: `npm ci` or `npm install`
- [ ] **Typecheck passes with zero errors**: `npx tsc --noEmit`
- [ ] **Production bundle builds cleanly**: `npm run build`
- [ ] **Preview works as expected**: `npm run preview`
- [ ] **Environment variable configured**: `VITE_API_URL` points to your active backend (e.g. `http://54.80.219.70:7860` or your deployed API gateway / load balancer).
- [ ] **Backend CORS enabled**: Ensure the FastAPI backend has configured CORS headers allowing your frontend domain (or `*` during initial deployment).
- [ ] **No sensitive tokens in frontend**: Verify no secret keys (e.g. `GROQ_API_KEY`, `HF_TOKEN`) are prefixed with `VITE_` or committed in the frontend codebase.

---

## 2. Environment Variables Reference

Vite inlines all environment variables prefixed with `VITE_` into the static JavaScript bundle at build time.

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `VITE_API_URL` | **Yes** (Prod) | `""` | Base URL of the deployed FastAPI backend (e.g., `http://54.80.219.70:7860` or `https://api.legalrag.example.com`). |
| `VITE_API_BASE_URL` | Optional | `""` | Alternative alias for `VITE_API_URL`. |
| `VITE_API_TIMEOUT_MS` | Optional | `120000` | Request timeout in milliseconds (default 120 seconds). |

> **Note**: Do not include a trailing slash in `VITE_API_URL`. The client automatically trims trailing slashes before appending `/query`.

---

## 3. Platform Deployment Guides

### Option A: Vercel (Recommended)

Vercel provides zero-configuration deployment for Vite projects.

#### Via Vercel CLI:
```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Login to Vercel
vercel login

# 3. Deploy from the frontend directory
cd frontend
vercel
```

#### Via Vercel Dashboard (Git Integration):
1. Push your repository to GitHub / GitLab.
2. Go to [Vercel Dashboard](https://vercel.com/new) and import the repository.
3. In project settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
4. Add Environment Variables:
   - `VITE_API_URL`: `http://54.80.219.70:7860` (or your backend URL)
   - `VITE_API_TIMEOUT_MS`: `120000`
5. Click **Deploy**.

#### SPA Routing on Vercel:
Create a `vercel.json` in `frontend/` (if using custom client routes):
```json
{
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

---

### Option B: Netlify

#### Via Netlify CLI:
```bash
# 1. Install Netlify CLI
npm install -g netlify-cli

# 2. Build the project
cd frontend
npm run build

# 3. Deploy
netlify deploy --prod --dir=dist
```

#### Via Netlify Dashboard:
1. Connect your Git repository on Netlify.
2. Configure build settings:
   - **Base directory**: `frontend`
   - **Build command**: `npm run build`
   - **Publish directory**: `frontend/dist`
3. Add Environment Variables in **Site settings > Environment variables**:
   - `VITE_API_URL`: `http://54.80.219.70:7860`
4. Deploy site.

#### Netlify Redirects (`_redirects`):
Create `frontend/public/_redirects` to handle SPA refreshes:
```
/*    /index.html   200
```

---

### Option C: Cloudflare Pages

1. In the Cloudflare Dashboard, go to **Workers & Pages** > **Create application** > **Pages** > **Connect to Git**.
2. Select your repository.
3. Configure build settings:
   - **Framework preset**: `Vite`
   - **Root directory**: `frontend`
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
4. Under **Environment variables**, set:
   - `VITE_API_URL` = `http://54.80.219.70:7860`
   - `NODE_VERSION` = `20`
5. Click **Save and Deploy**.

---

### Option D: Docker + Nginx (Self-Hosted / Container Platforms)

For deploying the frontend as a lightweight container on AWS ECS, GCP Cloud Run, Azure App Service, or VPS:

#### 1. Multi-Stage Dockerfile (`frontend/Dockerfile`):
```dockerfile
# Stage 1: Build static assets
FROM node:20-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Build argument for API URL
ARG VITE_API_URL=http://54.80.219.70:7860
ARG VITE_API_TIMEOUT_MS=120000
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_API_TIMEOUT_MS=$VITE_API_TIMEOUT_MS

RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 2. Nginx Configuration (`frontend/nginx.conf`):
```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip Compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, no-transform";
    }

    error_page 500 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
    }
}
```

#### 3. Build & Run:
```bash
docker build -t legalrag-frontend --build-arg VITE_API_URL=http://54.80.219.70:7860 .
docker run -d -p 80:80 legalrag-frontend
```

---

### Option E: AWS S3 + CloudFront

1. **Build the production bundle**:
   ```bash
   cd frontend
   VITE_API_URL=http://54.80.219.70:7860 npm run build
   ```
2. **Sync to S3 Bucket**:
   ```bash
   aws s3 sync dist/ s3://your-legalrag-frontend-bucket --delete
   ```
3. **Invalidate CloudFront Distribution**:
   ```bash
   aws cloudfront create-invalidation --distribution-id YOUR_DIST_ID --paths "/*"
   ```

---

## 4. Backend Connectivity & CORS Configuration (AWS EC2 t4.xlarge)

The LegalRAG production backend runs in a containerized environment on an **AWS EC2 instance** (`t4.xlarge`, 4 vCPUs, 16 GiB RAM).

### 1. EC2 Security Group Rules
Ensure the EC2 Security Group attached to your instance permits inbound traffic:
- **Port 7860** (FastAPI Default) or **Port 80/443** (if behind Nginx/ALB):
  - Type: `Custom TCP`
  - Port Range: `7860` (or `80`/`443`)
  - Source: `0.0.0.0/0` (or restricted to your frontend CDN / IP range)

### 2. Mixed Content Note (HTTPS vs HTTP)
- If your frontend SPA is hosted on an HTTPS domain (e.g., `https://legalrag.vercel.app` or Cloudflare Pages):
  - Browsers will block raw `http://<ec2-public-ip>:7860/query` requests due to **Mixed Content Security Policies**.
  - **Recommended Solutions**:
    1. **Nginx Reverse Proxy + Let's Encrypt (Certbot)** on the EC2 instance pointing a subdomain (e.g. `api.legalrag.yourdomain.com`) to `http://localhost:7860`.
    2. **AWS Application Load Balancer (ALB)** with an ACM SSL certificate forwarding HTTPS `:443` to EC2 target group port `:7860`.
    3. **Cloudflare Free SSL Proxy** tunneling to your EC2 instance.

### 3. FastAPI CORS Setup
Ensure the FastAPI backend running on EC2 allows requests from your frontend origin in `src/legalrag/api/main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify ["https://legalrag.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 5. Post-Deployment Smoke Test

Once deployed, perform the following end-to-end sanity tests:

1. **Initial Page Load**:
   - Visit the deployment URL.
   - Verify header badges show: `100k Judgments · 538k Chunks` and `Hybrid RRF + Reranker`.
   - Verify character counter reads `0 / 4,000 characters`.
2. **Populate Benchmark Query**:
   - Click one of the 6 benchmark query cards (e.g., *Section 482 CrPC quashing in matrimonial disputes*).
   - Ensure the textarea is populated with the question.
3. **Execute Inquiry**:
   - Click **Synthesize Precedents** (or press `Ctrl+Enter`).
   - Observe the live execution timer and 3-stage progress visualizer.
4. **Inspect Results**:
   - Verify the synthesis prose displays with inline badges (e.g. `[1]`, `[1 · CAL]`).
   - Click a citation badge — verify it highlights and scrolls to the source authority card.
   - Click a copy button on CNR and Chunk ID — verify the `Copied!` tooltip appears.
   - Verify the segmented execution telemetry bar displays `Retrieve`, `Rerank`, and `Groq LLM` latencies.
5. **New Inquiry**:
   - Click **← New Inquiry** and verify it returns to a clean search view.
