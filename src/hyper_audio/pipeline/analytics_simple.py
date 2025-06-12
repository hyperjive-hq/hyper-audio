"""Simplified pipeline analytics and monitoring."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from ..utils.logging_utils import get_logger
from .constants import JobStatus, DEFAULT_ANALYSIS_DAYS, RECENT_JOBS_LIMIT
from .models import PipelineState, JobSummary
from .checkpoint import StateManager

logger = get_logger("pipeline.analytics")


class PipelineAnalytics:
    """Simplified analytics for pipeline performance monitoring."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        """Initialize analytics with checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing pipeline checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.reports_dir = self.checkpoint_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed status of a pipeline job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        try:
            state_manager = StateManager(self.checkpoint_dir / job_id)
            state_data = state_manager.load_state()

            if not state_data:
                return {"job_id": job_id, "status": JobStatus.NOT_FOUND.value, "message": f"Job {job_id} not found"}

            state = PipelineState.from_dict(state_data)
            summary = JobSummary.from_state(state, len(state.stage_metrics))

            return {
                "job_id": job_id,
                "status": summary.status.value,
                "progress_percentage": summary.progress_percentage,
                "current_stage": summary.current_stage,
                "total_stages": summary.total_stages,
                "stages_completed": summary.stages_completed,
                "created_at": summary.created_at.isoformat(),
                "updated_at": summary.updated_at.isoformat(),
                "error_message": summary.error_message,
                "stage_metrics": [metric.to_dict() for metric in state.stage_metrics]
            }

        except Exception as e:
            logger.error(f"Failed to get job status for {job_id}: {e}")
            return {"job_id": job_id, "status": JobStatus.ERROR.value, "message": f"Failed to read job status: {e}"}

    def list_jobs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all pipeline jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job summaries
        """
        jobs = []

        for job_dir in self.checkpoint_dir.iterdir():
            if not job_dir.is_dir():
                continue

            try:
                status = self.get_job_status(job_dir.name)
                if status["status"] != JobStatus.NOT_FOUND.value:
                    jobs.append(status)
            except Exception as e:
                logger.warning(f"Failed to process job {job_dir.name}: {e}")
                continue

        # Sort by creation time (most recent first)
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        if limit:
            jobs = jobs[:limit]

        return jobs

    def get_recent_jobs(self, days: int = DEFAULT_ANALYSIS_DAYS, limit: int = RECENT_JOBS_LIMIT) -> List[Dict[str, Any]]:
        """Get recent jobs within specified time range.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of jobs to return
            
        Returns:
            List of recent job summaries
        """
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        all_jobs = self.list_jobs()

        recent_jobs = []
        for job in all_jobs:
            try:
                created_at = datetime.fromisoformat(job["created_at"])
                # Ensure created_at has timezone info, assume UTC if naive
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if created_at >= cutoff_date:
                    recent_jobs.append(job)
            except (ValueError, KeyError):
                continue

        return recent_jobs[:limit]

    def generate_summary_report(self, days: int = DEFAULT_ANALYSIS_DAYS) -> Dict[str, Any]:
        """Generate a summary report for recent jobs.
        
        Args:
            days: Number of days to include in analysis
            
        Returns:
            Summary report
        """
        recent_jobs = self.get_recent_jobs(days)

        if not recent_jobs:
            return {
                "period_days": days,
                "total_jobs": 0,
                "message": "No jobs found in the specified time range"
            }

        # Calculate summary statistics
        completed_jobs = [job for job in recent_jobs if job["status"] == JobStatus.COMPLETED.value]
        failed_jobs = [job for job in recent_jobs if job["status"] == JobStatus.FAILED.value]
        in_progress_jobs = [job for job in recent_jobs if job["status"] == JobStatus.IN_PROGRESS.value]

        # Stage failure analysis
        stage_failures = self._analyze_stage_failures(recent_jobs)

        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(completed_jobs)

        from datetime import timezone
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "summary": {
                "total_jobs": len(recent_jobs),
                "completed_jobs": len(completed_jobs),
                "failed_jobs": len(failed_jobs),
                "in_progress_jobs": len(in_progress_jobs),
                "success_rate": (len(completed_jobs) / len(recent_jobs)) * 100 if recent_jobs else 0
            },
            "stage_failures": stage_failures,
            "performance": performance_metrics,
            "recommendations": self._generate_recommendations(recent_jobs, stage_failures)
        }

        # Save report
        try:
            report_path = self.reports_dir / f"summary_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Summary report generated: {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save report file: {e}")

        return report

    def cleanup_old_reports(self, days: int = 30):
        """Clean up old report files.
        
        Args:
            days: Remove reports older than this many days
        """
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        for report_file in self.reports_dir.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(report_file.stat().st_mtime, tz=timezone.utc)
                if file_time < cutoff_date:
                    report_file.unlink()
                    logger.debug(f"Removed old report: {report_file}")
            except Exception as e:
                logger.warning(f"Failed to check report file {report_file}: {e}")

    def _analyze_stage_failures(self, jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stage failure patterns.
        
        Args:
            jobs: List of job data
            
        Returns:
            Stage failure analysis
        """
        stage_failures = {}
        total_failures = 0

        for job in jobs:
            for metric in job.get("stage_metrics", []):
                if metric.get("status") == "failed":
                    stage_name = metric.get("stage_name")
                    if stage_name:
                        stage_failures[stage_name] = stage_failures.get(stage_name, 0) + 1
                        total_failures += 1

        # Find most problematic stage
        most_failed_stage = None
        if stage_failures:
            most_failed_stage = max(stage_failures.items(), key=lambda x: x[1])

        return {
            "total_failures": total_failures,
            "failures_by_stage": stage_failures,
            "most_failed_stage": most_failed_stage[0] if most_failed_stage else None,
            "most_failed_count": most_failed_stage[1] if most_failed_stage else 0
        }

    def _calculate_performance_metrics(self, completed_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for completed jobs.
        
        Args:
            completed_jobs: List of completed job data
            
        Returns:
            Performance metrics
        """
        if not completed_jobs:
            return {"message": "No completed jobs found"}

        all_durations = []
        all_memory_usage = []
        stage_durations = {}

        for job in completed_jobs:
            job_duration = 0
            job_memory = 0

            for metric in job.get("stage_metrics", []):
                if metric.get("status") == "completed":
                    # Duration analysis
                    duration = metric.get("duration_seconds", 0) or 0
                    if duration and duration > 0:
                        job_duration += duration

                        stage_name = metric.get("stage_name")
                        if stage_name:
                            if stage_name not in stage_durations:
                                stage_durations[stage_name] = []
                            stage_durations[stage_name].append(duration)

                    # Memory analysis
                    memory = metric.get("memory_peak_gb", 0) or 0
                    if memory and memory > 0:
                        job_memory = max(job_memory, memory)

            if job_duration and job_duration > 0:
                all_durations.append(job_duration)
            if job_memory and job_memory > 0:
                all_memory_usage.append(job_memory)

        # Calculate averages for each stage
        avg_stage_durations = {}
        for stage_name, durations in stage_durations.items():
            avg_stage_durations[stage_name] = sum(durations) / len(durations)

        return {
            "avg_job_duration_seconds": sum(all_durations) / len(all_durations) if all_durations else 0,
            "max_job_duration_seconds": max(all_durations) if all_durations else 0,
            "min_job_duration_seconds": min(all_durations) if all_durations else 0,
            "avg_memory_usage_gb": sum(all_memory_usage) / len(all_memory_usage) if all_memory_usage else 0,
            "peak_memory_usage_gb": max(all_memory_usage) if all_memory_usage else 0,
            "avg_stage_durations": avg_stage_durations
        }

    def _generate_recommendations(self, jobs: List[Dict[str, Any]], stage_failures: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations.
        
        Args:
            jobs: List of job data
            stage_failures: Stage failure analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []

        if not jobs:
            return ["No recent jobs to analyze."]

        # Success rate recommendations
        success_rate = (len([j for j in jobs if j["status"] == "completed"]) / len(jobs)) * 100
        if success_rate < 80:
            recommendations.append(f"Low success rate ({success_rate:.1f}%). Focus on improving pipeline reliability.")

        # Stage failure recommendations
        if stage_failures.get("most_failed_stage"):
            stage = stage_failures["most_failed_stage"]
            count = stage_failures["most_failed_count"]
            recommendations.append(f"Focus on improving {stage} stage reliability ({count} failures detected).")

        # Performance recommendations
        completed_jobs = [job for job in jobs if job["status"] == "completed"]
        if completed_jobs:
            # Check for slow jobs
            for job in completed_jobs:
                total_duration = sum(
                    (metric.get("duration_seconds", 0) or 0)
                    for metric in job.get("stage_metrics", [])
                    if metric.get("status") == "completed"
                )
                if total_duration and total_duration > 1800:  # More than 30 minutes
                    recommendations.append("Consider performance optimization - some jobs are taking over 30 minutes.")
                    break

        # Memory recommendations
        for job in completed_jobs:
            for metric in job.get("stage_metrics", []):
                memory_usage = metric.get("memory_peak_gb", 0) or 0
                if memory_usage and memory_usage > 20:  # More than 20GB
                    recommendations.append("High memory usage detected. Consider memory optimization strategies.")
                    break
            else:
                continue
            break

        return recommendations if recommendations else ["Pipeline performance looks good. No specific recommendations."]
