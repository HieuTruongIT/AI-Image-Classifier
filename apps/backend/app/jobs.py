# apps/backend/app/jobs.py
"""
Async Job Queue for batch predictions.

Provides background processing for batch image classification with:
- Job submission and status polling
- Progress tracking
- Result retrieval
- Automatic cleanup of old jobs

For production at scale, replace with Redis + Celery or similar.
"""

from __future__ import annotations
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobResult:
    """Result for a single image in a batch."""
    index: int
    filename: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    top_k: Optional[List[Dict]] = None
    error: Optional[str] = None


@dataclass
class Job:
    """Async batch prediction job."""
    id: str
    status: JobStatus
    created_at: float
    total_images: int
    processed_images: int = 0
    results: List[JobResult] = field(default_factory=list)
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    @property
    def progress(self) -> float:
        if self.total_images == 0:
            return 0
        return self.processed_images / self.total_images
    
    @property
    def duration_ms(self) -> Optional[int]:
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return int((end - self.started_at) * 1000)
    
    def to_dict(self, include_results: bool = False) -> Dict[str, Any]:
        result = {
            "job_id": self.id,
            "status": self.status.value,
            "created_at": self.created_at,
            "total_images": self.total_images,
            "processed_images": self.processed_images,
            "progress": round(self.progress, 2),
            "duration_ms": self.duration_ms,
        }
        if self.error:
            result["error"] = self.error
        if include_results and self.status == JobStatus.COMPLETED:
            result["results"] = [
                {
                    "index": r.index,
                    "filename": r.filename,
                    "label": r.label,
                    "confidence": r.confidence,
                    "top_k": r.top_k,
                    "error": r.error,
                }
                for r in self.results
            ]
        return result


class JobQueue:
    """
    In-memory async job queue.
    
    For production, replace with:
    - Redis + Celery for distributed processing
    - AWS SQS + Lambda for serverless
    - Google Cloud Tasks for managed queues
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_jobs: int = 1000,
        job_ttl_seconds: float = 3600,  # 1 hour
    ):
        self.max_workers = max_workers
        self.max_jobs = max_jobs
        self.job_ttl_seconds = job_ttl_seconds
        
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def create_job(self, total_images: int) -> Job:
        """Create a new job and return it."""
        with self._lock:
            # Cleanup old jobs if at capacity
            if len(self._jobs) >= self.max_jobs:
                self._cleanup_old_jobs()
            
            job = Job(
                id=str(uuid.uuid4()),
                status=JobStatus.PENDING,
                created_at=time.time(),
                total_images=total_images,
            )
            self._jobs[job.id] = job
            return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        processed_images: Optional[int] = None,
        result: Optional[JobResult] = None,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        """Update job state."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            
            if status:
                job.status = status
                if status == JobStatus.PROCESSING and job.started_at is None:
                    job.started_at = time.time()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = time.time()
            
            if processed_images is not None:
                job.processed_images = processed_images
            
            if result:
                job.results.append(result)
                job.processed_images = len(job.results)
            
            if error:
                job.error = error
            
            return job
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                return True
            return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job (only completed/failed/cancelled)."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                del self._jobs[job_id]
                return True
            return False
    
    def list_jobs(self, limit: int = 50) -> List[Dict]:
        """List recent jobs."""
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )[:limit]
            return [j.to_dict() for j in jobs]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            status_counts = {}
            for job in self._jobs.values():
                status_counts[job.status.value] = status_counts.get(job.status.value, 0) + 1
            
            return {
                "total_jobs": len(self._jobs),
                "max_jobs": self.max_jobs,
                "max_workers": self.max_workers,
                "by_status": status_counts,
            }
    
    def _cleanup_old_jobs(self):
        """Remove expired jobs."""
        now = time.time()
        expired = [
            job_id for job_id, job in self._jobs.items()
            if (now - job.created_at) > self.job_ttl_seconds
            and job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
        ]
        for job_id in expired:
            del self._jobs[job_id]


# ─────────────────────────────────────────────────────────────
# Singleton Instance
# ─────────────────────────────────────────────────────────────
_job_queue: Optional[JobQueue] = None

def get_job_queue() -> JobQueue:
    """Get or create singleton job queue."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue
