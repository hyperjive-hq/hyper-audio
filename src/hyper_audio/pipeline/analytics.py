"""Pipeline analytics and monitoring tools."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging_utils import get_logger

logger = get_logger("pipeline.analytics")


class PipelineAnalytics:
    """Analytics and monitoring for pipeline performance."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        """Initialize analytics with checkpoint directory.
        
        Args:
            checkpoint_dir: Directory containing pipeline checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.reports_dir = self.checkpoint_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_job_report(self, job_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Detailed job report
        """
        job_dir = self.checkpoint_dir / job_id

        if not job_dir.exists():
            raise FileNotFoundError(f"Job {job_id} not found")

        # Load pipeline state
        state_path = job_dir / "pipeline_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"Pipeline state not found for job {job_id}")

        with open(state_path, 'r') as f:
            state_data = json.load(f)

        # Load failure report if exists
        failure_report = None
        failure_path = job_dir / "failure_report.json"
        if failure_path.exists():
            with open(failure_path, 'r') as f:
                failure_report = json.load(f)

        # Calculate summary statistics
        metrics = state_data.get('stage_metrics', [])
        total_duration = sum(m.get('duration_seconds', 0) for m in metrics if m.get('duration_seconds'))

        completed_stages = [m for m in metrics if m.get('status') == 'completed']
        failed_stages = [m for m in metrics if m.get('status') == 'failed']

        # Memory usage analysis
        memory_usage = [m.get('memory_peak_gb', 0) for m in metrics if m.get('memory_peak_gb')]
        max_memory = max(memory_usage) if memory_usage else 0

        # Checkpoint analysis
        checkpoint_sizes = [m.get('checkpoint_size_mb', 0) for m in metrics if m.get('checkpoint_size_mb')]
        total_checkpoint_size = sum(checkpoint_sizes)

        report = {
            "job_id": job_id,
            "generated_at": datetime.now().isoformat(),
            "status": state_data.get('status', 'unknown'),
            "summary": {
                "total_duration_seconds": total_duration,
                "total_duration_formatted": self._format_duration(total_duration),
                "stages_completed": len(completed_stages),
                "stages_failed": len(failed_stages),
                "total_stages": len(metrics),
                "success_rate": len(completed_stages) / len(metrics) * 100 if metrics else 0,
                "max_memory_usage_gb": max_memory,
                "total_checkpoint_size_mb": total_checkpoint_size
            },
            "stage_details": metrics,
            "performance_metrics": self._calculate_performance_metrics(metrics),
            "resource_usage": self._analyze_resource_usage(metrics),
            "failure_analysis": failure_report,
            "recommendations": self._generate_recommendations(metrics, failure_report)
        }

        # Save report
        report_path = self.reports_dir / f"{job_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Job report generated: {report_path}")
        return report

    def generate_aggregate_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate aggregate report across multiple jobs.
        
        Args:
            days: Number of days to include in analysis
            
        Returns:
            Aggregate analytics report
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        job_data = []

        # Collect data from all jobs
        for job_dir in self.checkpoint_dir.iterdir():
            if not job_dir.is_dir():
                continue

            state_path = job_dir / "pipeline_state.json"
            if not state_path.exists():
                continue

            try:
                with open(state_path, 'r') as f:
                    state_data = json.load(f)

                # Check if job is within date range
                created_at = datetime.fromisoformat(state_data.get('created_at', ''))
                if created_at < cutoff_date:
                    continue

                job_data.append({
                    "job_id": job_dir.name,
                    "created_at": created_at,
                    "metrics": state_data.get('stage_metrics', []),
                    "status": self._determine_job_status(state_data)
                })

            except Exception as e:
                logger.warning(f"Failed to process job {job_dir.name}: {e}")
                continue

        if not job_data:
            return {"message": "No jobs found in the specified time range"}

        # Generate aggregate statistics
        report = {
            "generated_at": datetime.now().isoformat(),
            "analysis_period_days": days,
            "total_jobs": len(job_data),
            "summary": self._calculate_aggregate_summary(job_data),
            "stage_performance": self._analyze_stage_performance(job_data),
            "failure_patterns": self._analyze_failure_patterns(job_data),
            "resource_trends": self._analyze_resource_trends(job_data),
            "recommendations": self._generate_aggregate_recommendations(job_data)
        }

        # Save report
        report_path = self.reports_dir / f"aggregate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Aggregate report generated: {report_path}")
        return report

    def create_performance_dashboard(self, job_ids: Optional[List[str]] = None) -> Path:
        """Create visual performance dashboard.
        
        Args:
            job_ids: Specific job IDs to include (all if None)
            
        Returns:
            Path to generated dashboard HTML file
        """
        # Collect data
        if job_ids is None:
            job_ids = [d.name for d in self.checkpoint_dir.iterdir() if d.is_dir()]

        all_metrics = []
        for job_id in job_ids:
            try:
                state_path = self.checkpoint_dir / job_id / "pipeline_state.json"
                if state_path.exists():
                    with open(state_path, 'r') as f:
                        state_data = json.load(f)

                    for metric in state_data.get('stage_metrics', []):
                        metric['job_id'] = job_id
                        all_metrics.append(metric)
            except Exception as e:
                logger.warning(f"Failed to load metrics for job {job_id}: {e}")

        if not all_metrics:
            raise ValueError("No metrics data found for dashboard generation")

        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Pipeline Performance Dashboard', fontsize=16)

        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(all_metrics)

        # 1. Stage Duration Distribution
        completed_df = df[df['status'] == 'completed']
        if not completed_df.empty:
            sns.boxplot(data=completed_df, x='stage_name', y='duration_seconds', ax=axes[0, 0])
            axes[0, 0].set_title('Stage Duration Distribution')
            axes[0, 0].set_xlabel('Stage')
            axes[0, 0].set_ylabel('Duration (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Memory Usage by Stage
        memory_df = df[df['memory_peak_gb'].notna()]
        if not memory_df.empty:
            sns.barplot(data=memory_df, x='stage_name', y='memory_peak_gb', ax=axes[0, 1])
            axes[0, 1].set_title('Peak Memory Usage by Stage')
            axes[0, 1].set_xlabel('Stage')
            axes[0, 1].set_ylabel('Memory (GB)')
            axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Success Rate by Stage
        stage_success_rates = df.groupby('stage_name')['status'].apply(
            lambda x: (x == 'completed').sum() / len(x) * 100
        ).reset_index()
        stage_success_rates.columns = ['stage_name', 'success_rate']

        sns.barplot(data=stage_success_rates, x='stage_name', y='success_rate', ax=axes[1, 0])
        axes[1, 0].set_title('Success Rate by Stage')
        axes[1, 0].set_xlabel('Stage')
        axes[1, 0].set_ylabel('Success Rate (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Retry Distribution
        retry_df = df[df['retry_count'] > 0]
        if not retry_df.empty:
            sns.countplot(data=retry_df, x='retry_count', ax=axes[1, 1])
            axes[1, 1].set_title('Retry Count Distribution')
            axes[1, 1].set_xlabel('Number of Retries')
            axes[1, 1].set_ylabel('Count')

        plt.tight_layout()

        # Save dashboard
        dashboard_path = self.reports_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Performance dashboard saved: {dashboard_path}")
        return dashboard_path

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def _calculate_performance_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics for a job."""
        completed_metrics = [m for m in metrics if m.get('status') == 'completed']

        if not completed_metrics:
            return {"message": "No completed stages found"}

        durations = [m.get('duration_seconds', 0) for m in completed_metrics]
        memory_usage = [m.get('memory_peak_gb', 0) for m in completed_metrics if m.get('memory_peak_gb')]

        return {
            "avg_stage_duration": sum(durations) / len(durations),
            "min_stage_duration": min(durations),
            "max_stage_duration": max(durations),
            "avg_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            "peak_memory_usage": max(memory_usage) if memory_usage else 0,
            "total_retries": sum(m.get('retry_count', 0) for m in metrics)
        }

    def _analyze_resource_usage(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource usage patterns."""
        memory_by_stage = {}
        duration_by_stage = {}

        for metric in metrics:
            stage_name = metric.get('stage_name')
            if stage_name and metric.get('status') == 'completed':
                if metric.get('memory_peak_gb'):
                    memory_by_stage[stage_name] = metric['memory_peak_gb']
                if metric.get('duration_seconds'):
                    duration_by_stage[stage_name] = metric['duration_seconds']

        return {
            "memory_by_stage": memory_by_stage,
            "duration_by_stage": duration_by_stage,
            "bottleneck_stage": max(duration_by_stage.items(), key=lambda x: x[1])[0] if duration_by_stage else None,
            "memory_intensive_stage": max(memory_by_stage.items(), key=lambda x: x[1])[0] if memory_by_stage else None
        }

    def _generate_recommendations(self, metrics: List[Dict[str, Any]], failure_report: Optional[Dict[str, Any]]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Analyze failures
        failed_stages = [m for m in metrics if m.get('status') == 'failed']
        if failed_stages:
            most_failed = max(set(m['stage_name'] for m in failed_stages),
                            key=lambda x: sum(1 for m in failed_stages if m['stage_name'] == x))
            recommendations.append(f"Focus on improving reliability of {most_failed} stage (most failures)")

        # Analyze performance
        completed_metrics = [m for m in metrics if m.get('status') == 'completed']
        if completed_metrics:
            durations = [(m['stage_name'], m.get('duration_seconds', 0)) for m in completed_metrics]
            slowest_stage = max(durations, key=lambda x: x[1])
            if slowest_stage[1] > 300:  # More than 5 minutes
                recommendations.append(f"Consider optimizing {slowest_stage[0]} stage (longest duration: {self._format_duration(slowest_stage[1])})")

        # Analyze memory usage
        memory_metrics = [m for m in metrics if m.get('memory_peak_gb')]
        if memory_metrics:
            max_memory = max(m['memory_peak_gb'] for m in memory_metrics)
            if max_memory > 20:  # More than 20GB
                recommendations.append(f"High memory usage detected ({max_memory:.1f}GB). Consider memory optimization.")

        # Analyze retries
        total_retries = sum(m.get('retry_count', 0) for m in metrics)
        if total_retries > 5:
            recommendations.append("High retry count suggests instability. Review error handling and resource management.")

        return recommendations if recommendations else ["Pipeline performed well. No specific optimizations needed."]

    def _determine_job_status(self, state_data: Dict[str, Any]) -> str:
        """Determine overall job status from state data."""
        current_stage = state_data.get('current_stage', 0)
        total_stages = 6  # Assuming 6 stages

        if current_stage >= total_stages:
            return "completed"
        elif any(m.get('status') == 'failed' for m in state_data.get('stage_metrics', [])):
            return "failed"
        else:
            return "in_progress"

    def _calculate_aggregate_summary(self, job_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate summary statistics."""
        completed_jobs = [job for job in job_data if job['status'] == 'completed']
        failed_jobs = [job for job in job_data if job['status'] == 'failed']

        all_durations = []
        all_memory_usage = []

        for job in job_data:
            for metric in job['metrics']:
                if metric.get('status') == 'completed':
                    if metric.get('duration_seconds'):
                        all_durations.append(metric['duration_seconds'])
                    if metric.get('memory_peak_gb'):
                        all_memory_usage.append(metric['memory_peak_gb'])

        return {
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "success_rate": len(completed_jobs) / len(job_data) * 100,
            "avg_job_duration": sum(all_durations) / len(all_durations) if all_durations else 0,
            "avg_memory_usage": sum(all_memory_usage) / len(all_memory_usage) if all_memory_usage else 0,
            "peak_memory_usage": max(all_memory_usage) if all_memory_usage else 0
        }

    def _analyze_stage_performance(self, job_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by stage across all jobs."""
        stage_stats = {}

        for job in job_data:
            for metric in job['metrics']:
                stage_name = metric.get('stage_name')
                if not stage_name:
                    continue

                if stage_name not in stage_stats:
                    stage_stats[stage_name] = {
                        'total_runs': 0,
                        'successful_runs': 0,
                        'failed_runs': 0,
                        'total_duration': 0,
                        'durations': [],
                        'memory_usage': []
                    }

                stats = stage_stats[stage_name]
                stats['total_runs'] += 1

                if metric.get('status') == 'completed':
                    stats['successful_runs'] += 1
                    if metric.get('duration_seconds'):
                        stats['total_duration'] += metric['duration_seconds']
                        stats['durations'].append(metric['duration_seconds'])
                elif metric.get('status') == 'failed':
                    stats['failed_runs'] += 1

                if metric.get('memory_peak_gb'):
                    stats['memory_usage'].append(metric['memory_peak_gb'])

        # Calculate final statistics
        for stage_name, stats in stage_stats.items():
            stats['success_rate'] = (stats['successful_runs'] / stats['total_runs'] * 100) if stats['total_runs'] > 0 else 0
            stats['avg_duration'] = (stats['total_duration'] / stats['successful_runs']) if stats['successful_runs'] > 0 else 0
            stats['avg_memory'] = sum(stats['memory_usage']) / len(stats['memory_usage']) if stats['memory_usage'] else 0

            # Remove raw data to keep report size manageable
            del stats['durations']
            del stats['memory_usage']

        return stage_stats

    def _analyze_failure_patterns(self, job_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure patterns across jobs."""
        failure_patterns = {
            'most_failed_stage': None,
            'common_error_types': {},
            'failure_by_stage': {},
            'retry_patterns': {}
        }

        all_failures = []

        for job in job_data:
            for metric in job['metrics']:
                if metric.get('status') == 'failed':
                    all_failures.append(metric)

        if not all_failures:
            return failure_patterns

        # Most failed stage
        stage_failures = {}
        for failure in all_failures:
            stage = failure.get('stage_name')
            if stage:
                stage_failures[stage] = stage_failures.get(stage, 0) + 1

        if stage_failures:
            failure_patterns['most_failed_stage'] = max(stage_failures.items(), key=lambda x: x[1])
            failure_patterns['failure_by_stage'] = stage_failures

        # Common error types
        error_types = {}
        for failure in all_failures:
            error_type = failure.get('error_type')
            if error_type:
                error_types[error_type] = error_types.get(error_type, 0) + 1

        failure_patterns['common_error_types'] = error_types

        # Retry patterns
        retry_counts = [f.get('retry_count', 0) for f in all_failures]
        if retry_counts:
            failure_patterns['retry_patterns'] = {
                'avg_retries': sum(retry_counts) / len(retry_counts),
                'max_retries': max(retry_counts),
                'jobs_requiring_retries': len([r for r in retry_counts if r > 0])
            }

        return failure_patterns

    def _analyze_resource_trends(self, job_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource usage trends over time."""
        # This is a simplified version - in production, you might want more sophisticated trend analysis
        recent_jobs = sorted(job_data, key=lambda x: x['created_at'], reverse=True)[:10]

        memory_trend = []
        duration_trend = []

        for job in recent_jobs:
            job_memory = []
            job_duration = []

            for metric in job['metrics']:
                if metric.get('status') == 'completed':
                    if metric.get('memory_peak_gb'):
                        job_memory.append(metric['memory_peak_gb'])
                    if metric.get('duration_seconds'):
                        job_duration.append(metric['duration_seconds'])

            if job_memory:
                memory_trend.append(max(job_memory))
            if job_duration:
                duration_trend.append(sum(job_duration))

        return {
            'memory_trend': memory_trend,
            'duration_trend': duration_trend,
            'avg_memory_recent': sum(memory_trend) / len(memory_trend) if memory_trend else 0,
            'avg_duration_recent': sum(duration_trend) / len(duration_trend) if duration_trend else 0
        }

    def _generate_aggregate_recommendations(self, job_data: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on aggregate data."""
        recommendations = []

        # Success rate analysis
        success_rate = len([job for job in job_data if job['status'] == 'completed']) / len(job_data) * 100
        if success_rate < 80:
            recommendations.append(f"Low success rate ({success_rate:.1f}%). Focus on improving pipeline reliability.")

        # Performance analysis
        all_durations = []
        for job in job_data:
            job_duration = sum(m.get('duration_seconds', 0) for m in job['metrics'] if m.get('status') == 'completed')
            if job_duration > 0:
                all_durations.append(job_duration)

        if all_durations:
            avg_duration = sum(all_durations) / len(all_durations)
            if avg_duration > 1800:  # More than 30 minutes
                recommendations.append(f"Average job duration is high ({self._format_duration(avg_duration)}). Consider performance optimization.")

        return recommendations if recommendations else ["Overall pipeline performance is good."]
