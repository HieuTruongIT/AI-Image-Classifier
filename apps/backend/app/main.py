# apps/backend/app/main.py
"""
VisionScale API - Production ML Inference Service

A production-ready image classification API with:
- ResNet18 backbone (ImageNet 1000 classes)
- TorchScript optimized inference
- Grad-CAM explainability
- Rate limiting
- Prometheus metrics
- Health checks
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image
from typing import Optional, List
import io
import os
import time
import logging

from .model_service import ModelService
from .gradcam import get_gradcam
from .rate_limit import get_rate_limiter, RateLimitConfig
from .cache import get_prediction_cache, compute_image_hash
from .jobs import get_job_queue, JobStatus, JobResult
from .auth import get_api_key_manager, optional_api_key, require_api_key, AUTH_ENABLED, APIKey
from .metrics import (
    now_ms, observe, render_prometheus,
    requests_total, errors_total, request_duration, inference_duration
)

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
APP_VERSION = "1.2.0"
MODEL_VERSION = "resnet18-imagenet-1k"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "20"))
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# FastAPI App with OpenAPI Documentation
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="VisionScale API",
    description="""
## Production ML Inference Service

VisionScale provides real-time image classification using a ResNet18 model 
trained on ImageNet (1000 classes).

### Features
- **Fast Inference**: TorchScript-optimized model, <200ms p50 latency
- **Explainability**: Grad-CAM visualization shows what the model "sees"
- **Production Ready**: Rate limiting, health checks, Prometheus metrics
- **1000 Classes**: Recognizes everyday objects, animals, vehicles, and more

### Quick Start
1. Upload an image to `/predict`
2. Get classification with confidence scores
3. Optionally request Grad-CAM heatmap via `/explain`
    """,
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Inference", "description": "Image classification endpoints"},
        {"name": "Batch", "description": "Batch processing and async jobs"},
        {"name": "Explainability", "description": "Model interpretation with Grad-CAM"},
        {"name": "Health", "description": "Service health and monitoring"},
    ],
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Remaining", "X-RateLimit-Limit"],
)

# ─────────────────────────────────────────────────────────────
# Pydantic Models (Response Schemas)
# ─────────────────────────────────────────────────────────────
class TopKPrediction(BaseModel):
    label: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")

class PredictResponse(BaseModel):
    label: str = Field(..., description="Top predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence of top prediction")
    top_k: List[TopKPrediction] = Field(..., description="Top-K predictions with scores")
    model_version: str = Field(..., description="Model version identifier")
    inference_ms: Optional[int] = Field(None, description="Inference time in milliseconds")
    cache_hit: bool = Field(False, description="Whether result was served from cache")

class ExplainResponse(BaseModel):
    label: str = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    heatmap_base64: str = Field(..., description="Base64 encoded Grad-CAM heatmap overlay")
    model_version: str = Field(..., description="Model version identifier")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")

class VersionResponse(BaseModel):
    api_version: str = Field(..., description="API version")
    model_version: str = Field(..., description="Model version")
    features: List[str] = Field(..., description="Available features")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")

# ─────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────
def validate_and_load_image(body: bytes) -> Image.Image:
    """Validate and load image from bytes."""
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_UPLOAD_MB} MB).")
    try:
        img = Image.open(io.BytesIO(body))
        img.verify()
        img = Image.open(io.BytesIO(body)).convert("RGB")
        if img.width > 8000 or img.height > 8000:
            raise HTTPException(status_code=400, detail="Image dimensions too large (max 8000px).")
        return img
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

# Singleton model service
_service: ModelService | None = None
def service() -> ModelService:
    global _service
    if _service is None:
        logger.info("Loading model service...")
        _service = ModelService.create()
        logger.info("Model service loaded successfully")
    return _service

# ─────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────
@app.middleware("http")
async def timing_and_logging_mw(request: Request, call_next):
    """Timing, logging, and rate limiting middleware."""
    t0 = now_ms()
    route = request.url.path
    method = request.method
    status = 500
    
    # Rate limiting (skip for health/metrics)
    if RATE_LIMIT_ENABLED and route not in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
        try:
            limits = get_rate_limiter().check_rate_limit(request)
        except HTTPException as e:
            logger.warning(f"Rate limit exceeded for {request.client.host if request.client else 'unknown'}")
            return JSONResponse(status_code=e.status_code, content=e.detail)
    
    try:
        resp = await call_next(request)
        status = resp.status_code
        return resp
    except HTTPException as he:
        status = he.status_code
        raise
    except Exception:
        status = 500
        raise
    finally:
        dur = now_ms() - t0
        requests_total[(route, method, status)] += 1
        observe(request_duration, route, dur)
        if status >= 400:
            errors_total[(route, status)] += 1
        
        # Structured logging
        logger.info(f'{{"route":"{route}","method":"{method}","status":{status},"duration_ms":{dur}}}')

# ─────────────────────────────────────────────────────────────
# Health & Info Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Root endpoint with API information."""
    return {
        "service": "VisionScale API",
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": ["/predict", "/explain", "/health", "/version", "/metrics"],
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint for load balancers and monitoring."""
    try:
        # Try to access the model to verify it's loaded
        _ = service()
        return {"status": "ok", "model_loaded": True}
    except Exception:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False}
        )

@app.get("/version", response_model=VersionResponse, tags=["Health"])
def version():
    """Get API and model version information."""
    return {
        "api_version": APP_VERSION,
        "model_version": MODEL_VERSION,
        "features": [
            "image_classification",
            "gradcam_explainability",
            "top_k_predictions",
            "batch_prediction",
            "async_jobs",
            "prometheus_metrics",
            "rate_limiting",
            "response_caching",
            "api_key_auth",
        ],
    }

@app.get("/cache/stats", tags=["Health"])
def cache_stats():
    """Get cache statistics for monitoring."""
    stats = get_prediction_cache().get_stats()
    return {
        "enabled": CACHE_ENABLED,
        **stats,
    }

@app.post("/cache/clear", tags=["Health"])
def cache_clear():
    """Clear the prediction cache (admin endpoint)."""
    get_prediction_cache().clear()
    return {"message": "Cache cleared", "success": True}

@app.get("/metrics", tags=["Health"])
def metrics():
    """Prometheus-compatible metrics endpoint."""
    text = render_prometheus()
    return Response(content=text, media_type="text/plain; version=0.0.4")

# ─────────────────────────────────────────────────────────────
# Inference Endpoints
# ─────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        429: {"description": "Rate limit exceeded"},
    },
    tags=["Inference"],
    summary="Classify an image",
    description="""
Upload an image to get classification predictions.

Returns the top predicted class along with confidence scores for the top-K predictions.
Supports JPEG, PNG, WebP, and other common image formats.
    """,
)
async def predict(
    file: UploadFile = File(..., description="Image file to classify"),
    top_k: int = Query(default=5, ge=1, le=10, description="Number of top predictions to return"),
):
    """Classify an uploaded image and return top-K predictions."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image/* uploads are supported.")
    
    body = await file.read()
    
    # Check cache first (if enabled)
    cache_hit = False
    cache_key = None
    if CACHE_ENABLED:
        cache_key = f"{compute_image_hash(body)}:k={top_k}"
        cached = get_prediction_cache().get(cache_key)
        if cached:
            logger.info(f'{{"event":"cache_hit","cache_key":"{cache_key[:16]}..."}}')
            return {**cached, "cache_hit": True, "inference_ms": 0}
    
    img = validate_and_load_image(body)

    t0 = now_ms()
    topk = service().predict_image(img, k=top_k)
    infer_ms = now_ms() - t0
    observe(inference_duration, MODEL_VERSION, infer_ms)

    result = {
        "label": topk[0]["label"],
        "confidence": topk[0]["confidence"],
        "top_k": topk,
        "model_version": MODEL_VERSION,
        "inference_ms": infer_ms,
        "cache_hit": False,
    }
    
    # Store in cache
    if CACHE_ENABLED and cache_key:
        cache_result = {k: v for k, v in result.items() if k not in ("cache_hit", "inference_ms")}
        get_prediction_cache().set(cache_key, cache_result)
    
    return result

@app.post(
    "/explain",
    response_model=ExplainResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
    },
    tags=["Explainability"],
    summary="Get Grad-CAM explanation",
    description="""
Generate a Grad-CAM heatmap showing which parts of the image 
influenced the model's prediction.

The heatmap is returned as a base64-encoded PNG image overlaid 
on the original image. Red/yellow regions indicate areas the 
model focused on.
    """,
)
async def explain(
    file: UploadFile = File(..., description="Image file to explain"),
    alpha: float = Query(default=0.5, ge=0.1, le=0.9, description="Heatmap overlay transparency"),
):
    """Generate Grad-CAM explanation for an image prediction."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image/* uploads are supported.")
    
    body = await file.read()
    img = validate_and_load_image(body)
    
    t0 = now_ms()
    heatmap_b64, pred_class, confidence = get_gradcam().generate_overlay(img, alpha=alpha)
    infer_ms = now_ms() - t0
    
    # Get label for predicted class
    labels = service().categories
    label = labels[pred_class] if pred_class < len(labels) else f"class_{pred_class}"
    
    logger.info(f'{{"endpoint":"/explain","class":"{label}","confidence":{confidence:.3f},"duration_ms":{infer_ms}}}')
    
    return {
        "label": label,
        "confidence": confidence,
        "heatmap_base64": heatmap_b64,
        "model_version": MODEL_VERSION,
    }

# ─────────────────────────────────────────────────────────────
# Sample Images Endpoint
# ─────────────────────────────────────────────────────────────
SAMPLE_IMAGES = [
    {
        "id": "dog",
        "name": "Golden Retriever",
        "url": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",
        "category": "animal",
    },
    {
        "id": "cat",
        "name": "Tabby Cat",
        "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400",
        "category": "animal",
    },
    {
        "id": "car",
        "name": "Sports Car",
        "url": "https://images.unsplash.com/photo-1544636331-e26879cd4d9b?w=400",
        "category": "vehicle",
    },
    {
        "id": "bird",
        "name": "Hummingbird",
        "url": "https://images.unsplash.com/photo-1555169062-013468b47731?w=400",
        "category": "animal",
    },
    {
        "id": "flower",
        "name": "Sunflower",
        "url": "https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=400",
        "category": "plant",
    },
    {
        "id": "food",
        "name": "Pizza",
        "url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=400",
        "category": "food",
    },
]

@app.get("/samples", tags=["Inference"])
def get_sample_images():
    """Get list of sample images for testing."""
    return {"samples": SAMPLE_IMAGES}


# ─────────────────────────────────────────────────────────────
# Batch Prediction Endpoints
# ─────────────────────────────────────────────────────────────
class BatchPredictionResult(BaseModel):
    index: int = Field(..., description="Index of image in batch")
    filename: str = Field(..., description="Original filename")
    label: Optional[str] = Field(None, description="Predicted class")
    confidence: Optional[float] = Field(None, description="Confidence score")
    top_k: Optional[List[TopKPrediction]] = Field(None, description="Top-K predictions")
    error: Optional[str] = Field(None, description="Error message if prediction failed")

class BatchPredictResponse(BaseModel):
    total: int = Field(..., description="Total images in batch")
    successful: int = Field(..., description="Successfully processed images")
    failed: int = Field(..., description="Failed images")
    results: List[BatchPredictionResult] = Field(..., description="Prediction results")
    total_ms: int = Field(..., description="Total processing time in ms")

@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    tags=["Batch"],
    summary="Batch classify multiple images",
    description=f"""
Upload multiple images for batch classification.

- Maximum {MAX_BATCH_SIZE} images per batch
- Each image is processed synchronously
- Returns results for all images, including any errors
    """,
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Image files to classify"),
    top_k: int = Query(default=5, ge=1, le=10, description="Number of top predictions per image"),
):
    """Classify multiple images in a single request."""
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE} images"
        )
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    t0 = now_ms()
    results = []
    successful = 0
    failed = 0
    
    for idx, file in enumerate(files):
        result = BatchPredictionResult(
            index=idx,
            filename=file.filename or f"image_{idx}",
        )
        
        try:
            if not file.content_type or not file.content_type.startswith("image/"):
                raise ValueError("Not an image file")
            
            body = await file.read()
            img = validate_and_load_image(body)
            
            topk = service().predict_image(img, k=top_k)
            
            result.label = topk[0]["label"]
            result.confidence = topk[0]["confidence"]
            result.top_k = topk
            successful += 1
            
        except Exception as e:
            result.error = str(e)
            failed += 1
        
        results.append(result)
    
    total_ms = now_ms() - t0
    
    logger.info(f'{{"endpoint":"/predict/batch","total":{len(files)},"successful":{successful},"failed":{failed},"duration_ms":{total_ms}}}')
    
    return {
        "total": len(files),
        "successful": successful,
        "failed": failed,
        "results": results,
        "total_ms": total_ms,
    }


# ─────────────────────────────────────────────────────────────
# Async Job Endpoints
# ─────────────────────────────────────────────────────────────
class JobSubmitResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    total_images: int = Field(..., description="Number of images to process")
    poll_url: str = Field(..., description="URL to poll for status")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., description="Progress (0-1)")
    total_images: int = Field(..., description="Total images")
    processed_images: int = Field(..., description="Processed images")
    duration_ms: Optional[int] = Field(None, description="Processing time so far")

@app.post(
    "/jobs/submit",
    response_model=JobSubmitResponse,
    tags=["Batch"],
    summary="Submit async batch job",
    description="""
Submit images for asynchronous batch processing.

Returns a job ID that can be polled for status and results.
Use this for large batches that would timeout with synchronous `/predict/batch`.
    """,
)
async def submit_job(
    files: List[UploadFile] = File(..., description="Image files to process"),
    top_k: int = Query(default=5, ge=1, le=10, description="Number of top predictions per image"),
):
    """Submit a batch prediction job for async processing."""
    if len(files) > MAX_BATCH_SIZE * 5:  # Allow larger batches for async
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {MAX_BATCH_SIZE * 5} images for async jobs"
        )
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create job
    queue = get_job_queue()
    job = queue.create_job(total_images=len(files))
    
    # Process in background
    import asyncio
    asyncio.create_task(_process_job(job.id, files, top_k))
    
    logger.info(f'{{"event":"job_submitted","job_id":"{job.id}","total_images":{len(files)}}}')
    
    return {
        "job_id": job.id,
        "status": job.status.value,
        "total_images": len(files),
        "poll_url": f"/jobs/{job.id}",
    }


async def _process_job(job_id: str, files: List[UploadFile], top_k: int):
    """Background task to process job."""
    queue = get_job_queue()
    queue.update_job(job_id, status=JobStatus.PROCESSING)
    
    try:
        for idx, file in enumerate(files):
            # Check if job was cancelled
            job = queue.get_job(job_id)
            if job is None or job.status == JobStatus.CANCELLED:
                return
            
            result = JobResult(
                index=idx,
                filename=file.filename or f"image_{idx}",
            )
            
            try:
                body = await file.read()
                img = validate_and_load_image(body)
                topk = service().predict_image(img, k=top_k)
                
                result.label = topk[0]["label"]
                result.confidence = topk[0]["confidence"]
                result.top_k = topk
                
            except Exception as e:
                result.error = str(e)
            
            queue.update_job(job_id, result=result)
        
        queue.update_job(job_id, status=JobStatus.COMPLETED)
        logger.info(f'{{"event":"job_completed","job_id":"{job_id}"}}')
        
    except Exception as e:
        queue.update_job(job_id, status=JobStatus.FAILED, error=str(e))
        logger.error(f'{{"event":"job_failed","job_id":"{job_id}","error":"{str(e)}"}}')


@app.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Batch"],
    summary="Get job status",
)
def get_job_status(job_id: str):
    """Get the status of an async job."""
    job = get_job_queue().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.id,
        "status": job.status.value,
        "progress": job.progress,
        "total_images": job.total_images,
        "processed_images": job.processed_images,
        "duration_ms": job.duration_ms,
    }


@app.get(
    "/jobs/{job_id}/results",
    tags=["Batch"],
    summary="Get job results",
)
def get_job_results(job_id: str):
    """Get the results of a completed job."""
    job = get_job_queue().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.PENDING:
        raise HTTPException(status_code=400, detail="Job not started yet")
    
    if job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Job still processing")
    
    return job.to_dict(include_results=True)


@app.delete(
    "/jobs/{job_id}",
    tags=["Batch"],
    summary="Cancel or delete a job",
)
def cancel_job(job_id: str):
    """Cancel a pending/processing job or delete a completed job."""
    queue = get_job_queue()
    job = queue.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
        queue.cancel_job(job_id)
        return {"message": "Job cancelled", "job_id": job_id}
    else:
        queue.delete_job(job_id)
        return {"message": "Job deleted", "job_id": job_id}


@app.get("/jobs", tags=["Batch"], summary="List jobs")
def list_jobs(limit: int = Query(default=20, ge=1, le=100)):
    """List recent jobs."""
    return {
        "jobs": get_job_queue().list_jobs(limit=limit),
        "stats": get_job_queue().get_stats(),
    }


# ─────────────────────────────────────────────────────────────
# API Key Management Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/auth/status", tags=["Health"])
def auth_status():
    """Check if API key authentication is enabled."""
    return {
        "auth_enabled": AUTH_ENABLED,
        "stats": get_api_key_manager().get_stats() if AUTH_ENABLED else None,
    }


@app.post("/auth/keys", tags=["Health"])
def create_api_key(
    name: str = Query(..., description="Name for the API key"),
    rate_limit_rpm: int = Query(60, description="Rate limit (requests per minute)"),
):
    """
    Create a new API key (admin endpoint).
    
    Returns the plaintext key - save it securely, it won't be shown again!
    """
    key = get_api_key_manager().create_key(name, rate_limit_rpm)
    return {
        "message": "API key created successfully",
        "key": key,
        "name": name,
        "warning": "Save this key securely - it won't be shown again!",
    }


@app.get("/auth/keys", tags=["Health"])
def list_api_keys():
    """List all API keys (without revealing the actual keys)."""
    return {
        "keys": get_api_key_manager().list_keys(),
        "stats": get_api_key_manager().get_stats(),
    }


@app.delete("/auth/keys/{key_hash}", tags=["Health"])
def revoke_api_key(key_hash: str):
    """Revoke an API key by its hash prefix."""
    # Find the full hash matching the prefix
    manager = get_api_key_manager()
    for full_hash in manager._keys.keys():
        if full_hash.startswith(key_hash):
            manager.revoke_key(full_hash)
            return {"message": "API key revoked", "key_hash": key_hash}
    
    raise HTTPException(status_code=404, detail="API key not found")
