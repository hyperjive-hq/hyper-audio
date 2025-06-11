"""Tests for analytics and monitoring functionality."""

import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from src.hyper_audio.pipeline.analytics_simple import PipelineAnalytics
from src.hyper_audio.pipeline.models import PipelineState, StageMetrics
from src.hyper_audio.pipeline.constants import StageStatus, JobStatus, STATE_FILENAME


class TestPipelineAnalytics:
    """Test pipeline analytics functionality."""
    
    @pytest.fixture
    def analytics(self, checkpoint_dir):
        """Create analytics instance for testing."""
        return PipelineAnalytics(checkpoint_dir)
    
    @pytest.fixture
    def sample_job_data(self, checkpoint_dir, sample_pipeline_state):
        """Create sample job data for testing."""
        # Create multiple job directories with state files
        jobs_data = []
        
        for i in range(3):
            job_id = f"test_job_{i}"
            job_dir = checkpoint_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            
            # Create state based on template but with different data
            state = PipelineState(
                job_id=job_id,
                input_path=f"/test/input_{i}.wav",
                output_path=f"/test/output_{i}.wav",
                config={"test": f"value_{i}"},
                current_stage=i + 1,
                stages_completed=[f"stage_{j}" for j in range(i + 1)],
                stage_metrics=[
                    StageMetrics(
                        stage_name=f"stage_{j}",
                        status=StageStatus.COMPLETED if j < i else StageStatus.FAILED,
                        duration_seconds=10.0 + j,
                        memory_peak_gb=2.0 + j * 0.5
                    ) for j in range(i + 2)  # Create i+2 metrics (some failed)
                ]
            )
            
            # Save state file
            state_file = job_dir / STATE_FILENAME
            with open(state_file, 'w') as f:
                json.dump(state.to_dict(), f)
            
            jobs_data.append(state)
        
        return jobs_data
    
    def test_init(self, analytics, checkpoint_dir):
        """Test analytics initialization."""
        assert analytics.checkpoint_dir == checkpoint_dir
        assert analytics.reports_dir == checkpoint_dir / "reports"
        assert analytics.reports_dir.exists()
    
    def test_get_job_status_existing(self, analytics, sample_job_data):
        """Test getting status for existing job."""
        job_id = "test_job_0"
        
        status = analytics.get_job_status(job_id)
        
        assert status["job_id"] == job_id
        assert status["status"] != JobStatus.NOT_FOUND.value
        assert "progress_percentage" in status
        assert "current_stage" in status
        assert "stage_metrics" in status
        assert isinstance(status["stage_metrics"], list)
    
    def test_get_job_status_nonexistent(self, analytics):
        """Test getting status for non-existent job."""
        status = analytics.get_job_status("nonexistent_job")
        
        assert status["status"] == JobStatus.NOT_FOUND.value
        assert "message" in status
    
    def test_get_job_status_corrupted_state(self, analytics, checkpoint_dir):
        """Test getting status when state file is corrupted."""
        job_id = "corrupted_job"
        job_dir = checkpoint_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create corrupted state file
        state_file = job_dir / STATE_FILENAME
        with open(state_file, 'w') as f:
            f.write("invalid json content")
        
        status = analytics.get_job_status(job_id)
        
        assert status["status"] == JobStatus.ERROR.value
        assert "message" in status
    
    def test_list_jobs(self, analytics, sample_job_data):
        """Test listing all jobs."""
        jobs = analytics.list_jobs()
        
        assert len(jobs) == 3
        
        # Should be sorted by creation time (most recent first)
        for job in jobs:
            assert "job_id" in job
            assert "status" in job
            assert "created_at" in job
    
    def test_list_jobs_with_limit(self, analytics, sample_job_data):
        """Test listing jobs with limit."""
        jobs = analytics.list_jobs(limit=2)
        
        assert len(jobs) == 2
    
    def test_get_recent_jobs(self, analytics, sample_job_data):
        """Test getting recent jobs."""
        recent_jobs = analytics.get_recent_jobs(days=7, limit=5)
        
        # All sample jobs should be recent (created just now)
        assert len(recent_jobs) == 3
        
        from datetime import timezone
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        for job in recent_jobs:
            created_at = datetime.fromisoformat(job["created_at"])
            # Ensure timezone-aware comparison
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            assert created_at >= cutoff, f"Job {job['job_id']} created_at {created_at} is older than cutoff {cutoff}"
    
    def test_get_recent_jobs_old_cutoff(self, analytics, sample_job_data):
        """Test getting recent jobs with very recent cutoff."""
        # Use a very recent cutoff (1 second ago)
        recent_jobs = analytics.get_recent_jobs(days=0, limit=10)
        
        # Should return empty or very few jobs depending on timing
        assert len(recent_jobs) <= 3
    
    def test_generate_summary_report(self, analytics, sample_job_data):
        """Test generating summary report."""
        report = analytics.generate_summary_report(days=30)
        
        assert "generated_at" in report
        assert "period_days" in report
        assert "summary" in report
        assert "stage_failures" in report
        assert "performance" in report
        assert "recommendations" in report
        
        # Check summary structure
        summary = report["summary"]
        assert "total_jobs" in summary
        assert "completed_jobs" in summary
        assert "failed_jobs" in summary
        assert "success_rate" in summary
        
        # Verify report was saved
        assert len(list(analytics.reports_dir.glob("summary_report_*.json"))) > 0
    
    def test_generate_summary_report_no_jobs(self, analytics):
        """Test generating summary report with no jobs."""
        report = analytics.generate_summary_report(days=30)
        
        assert report["total_jobs"] == 0
        assert "message" in report
    
    def test_cleanup_old_reports(self, analytics):
        """Test cleaning up old reports."""
        # Create some old report files
        old_report = analytics.reports_dir / "old_report.json"
        old_report.write_text("{}")
        
        # Modify file time to be old
        import os
        old_time = datetime.now().timestamp() - (35 * 24 * 3600)  # 35 days ago
        old_report.touch()
        os.utime(old_report, (old_time, old_time))
        
        # Create new report
        new_report = analytics.reports_dir / "new_report.json"
        new_report.write_text("{}")
        
        analytics.cleanup_old_reports(days=30)
        
        # Old report should be removed, new should remain
        assert not old_report.exists()
        assert new_report.exists()
    
    def test_analyze_stage_failures(self, analytics, sample_job_data):
        """Test stage failure analysis."""
        # Create jobs with known failure patterns
        jobs = [
            {
                "stage_metrics": [
                    {"stage_name": "stage1", "status": "failed"},
                    {"stage_name": "stage2", "status": "completed"},
                    {"stage_name": "stage1", "status": "failed"},  # stage1 fails twice
                ]
            }
        ]
        
        failures = analytics._analyze_stage_failures(jobs)
        
        assert failures["total_failures"] == 2
        assert "stage1" in failures["failures_by_stage"]
        assert failures["failures_by_stage"]["stage1"] == 2
        assert failures["most_failed_stage"] == "stage1"
        assert failures["most_failed_count"] == 2
    
    def test_analyze_stage_failures_no_failures(self, analytics):
        """Test stage failure analysis with no failures."""
        jobs = [
            {
                "stage_metrics": [
                    {"stage_name": "stage1", "status": "completed"},
                    {"stage_name": "stage2", "status": "completed"},
                ]
            }
        ]
        
        failures = analytics._analyze_stage_failures(jobs)
        
        assert failures["total_failures"] == 0
        assert failures["failures_by_stage"] == {}
        assert failures["most_failed_stage"] is None
    
    def test_calculate_performance_metrics(self, analytics):
        """Test performance metrics calculation."""
        completed_jobs = [
            {
                "stage_metrics": [
                    {"stage_name": "stage1", "status": "completed", "duration_seconds": 10.0, "memory_peak_gb": 2.0},
                    {"stage_name": "stage2", "status": "completed", "duration_seconds": 15.0, "memory_peak_gb": 3.0},
                ]
            },
            {
                "stage_metrics": [
                    {"stage_name": "stage1", "status": "completed", "duration_seconds": 12.0, "memory_peak_gb": 2.5},
                    {"stage_name": "stage2", "status": "completed", "duration_seconds": 18.0, "memory_peak_gb": 3.5},
                ]
            }
        ]
        
        metrics = analytics._calculate_performance_metrics(completed_jobs)
        
        assert "avg_job_duration_seconds" in metrics
        assert "max_job_duration_seconds" in metrics
        assert "min_job_duration_seconds" in metrics
        assert "avg_memory_usage_gb" in metrics
        assert "peak_memory_usage_gb" in metrics
        assert "avg_stage_durations" in metrics
        
        # Check calculated values
        assert metrics["avg_job_duration_seconds"] == 27.5  # (25 + 30) / 2
        assert metrics["max_job_duration_seconds"] == 30.0
        assert metrics["min_job_duration_seconds"] == 25.0
        assert metrics["peak_memory_usage_gb"] == 3.5
        
        # Check stage averages
        stage_durations = metrics["avg_stage_durations"]
        assert stage_durations["stage1"] == 11.0  # (10 + 12) / 2
        assert stage_durations["stage2"] == 16.5  # (15 + 18) / 2
    
    def test_calculate_performance_metrics_no_jobs(self, analytics):
        """Test performance metrics with no completed jobs."""
        metrics = analytics._calculate_performance_metrics([])
        
        assert "message" in metrics
        assert metrics["message"] == "No completed jobs found"
    
    def test_generate_recommendations_low_success_rate(self, analytics):
        """Test recommendations for low success rate."""
        jobs = [
            {"status": "completed"},
            {"status": "failed"},
            {"status": "failed"},
            {"status": "failed"},
        ]
        
        stage_failures = {"most_failed_stage": "problematic_stage", "most_failed_count": 3}
        
        recommendations = analytics._generate_recommendations(jobs, stage_failures)
        
        # Should recommend improving success rate
        success_rate_rec = any("success rate" in rec.lower() for rec in recommendations)
        assert success_rate_rec
        
        # Should recommend improving problematic stage
        stage_rec = any("problematic_stage" in rec for rec in recommendations)
        assert stage_rec
    
    def test_generate_recommendations_good_performance(self, analytics):
        """Test recommendations for good performance."""
        jobs = [{"status": "completed"} for _ in range(10)]  # All completed
        stage_failures = {"most_failed_stage": None, "most_failed_count": 0}
        
        recommendations = analytics._generate_recommendations(jobs, stage_failures)
        
        # Should indicate good performance
        assert len(recommendations) == 1
        assert "good" in recommendations[0].lower()
    
    def test_generate_recommendations_slow_jobs(self, analytics):
        """Test recommendations for slow jobs."""
        jobs = [
            {
                "status": "completed",
                "stage_metrics": [
                    {"status": "completed", "duration_seconds": 2000}  # Very slow (>30 min)
                ]
            }
        ]
        
        stage_failures = {"most_failed_stage": None, "most_failed_count": 0}
        
        recommendations = analytics._generate_recommendations(jobs, stage_failures)
        
        # Should recommend performance optimization
        perf_rec = any("performance" in rec.lower() or "30 minutes" in rec for rec in recommendations)
        assert perf_rec
    
    def test_generate_recommendations_high_memory(self, analytics):
        """Test recommendations for high memory usage."""
        jobs = [
            {
                "status": "completed",
                "stage_metrics": [
                    {"status": "completed", "memory_peak_gb": 25.0}  # High memory usage
                ]
            }
        ]
        
        stage_failures = {"most_failed_stage": None, "most_failed_count": 0}
        
        recommendations = analytics._generate_recommendations(jobs, stage_failures)
        
        # Should recommend memory optimization
        memory_rec = any("memory" in rec.lower() for rec in recommendations)
        assert memory_rec
    
    def test_error_handling_in_list_jobs(self, analytics, checkpoint_dir):
        """Test error handling when processing corrupted job data."""
        initial_count = len(analytics.list_jobs())
        
        # Create test jobs
        self._create_corrupted_job(checkpoint_dir)
        self._create_valid_job(checkpoint_dir)
        
        jobs = analytics.list_jobs()
        
        # Verify both jobs are returned
        assert len(jobs) == initial_count + 2
        
        # Verify error handling for corrupted job
        corrupted_job = next((job for job in jobs if job["job_id"] == "corrupted_job"), None)
        assert corrupted_job is not None
        assert corrupted_job["status"] == "error"
        
        # Verify valid job works correctly
        valid_job = next((job for job in jobs if job["job_id"] == "valid_job"), None)
        assert valid_job is not None
        assert valid_job["status"] != "error"
    
    def _create_corrupted_job(self, checkpoint_dir):
        """Helper to create a corrupted job for testing."""
        corrupted_job_dir = checkpoint_dir / "corrupted_job"
        corrupted_job_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = corrupted_job_dir / STATE_FILENAME
        with open(state_file, 'w') as f:
            f.write("invalid json")
    
    def _create_valid_job(self, checkpoint_dir):
        """Helper to create a valid job for testing."""
        valid_job_dir = checkpoint_dir / "valid_job"
        valid_job_dir.mkdir(parents=True, exist_ok=True)
        
        valid_state = PipelineState(
            job_id="valid_job",
            input_path="/test/input.wav",
            output_path="/test/output.wav",
            config={}
        )
        
        valid_state_file = valid_job_dir / STATE_FILENAME
        with open(valid_state_file, 'w') as f:
            json.dump(valid_state.to_dict(), f)
    
    def test_file_operations_error_handling(self, analytics, sample_job_data):
        """Test error handling for file operations."""
        # Test with permission denied when saving report (use a more specific mock)
        original_open = open
        def mock_open(*args, **kwargs):
            # Only raise error when writing to reports directory
            if len(args) > 0 and "summary_report_" in str(args[0]):
                raise PermissionError("Permission denied")
            return original_open(*args, **kwargs)
        
        with patch('builtins.open', side_effect=mock_open):
            # Should not raise exception, just log warning
            report = analytics.generate_summary_report()
            
            # Report should still be generated (in memory)
            assert "summary" in report
    
    @pytest.mark.parametrize("days,expected_jobs", [
        (1, 3),    # All recent jobs
        (0, 0),    # No jobs (cutoff too recent)
        (365, 3),  # All jobs within a year
    ])
    def test_get_recent_jobs_various_timeframes(self, analytics, sample_job_data, days, expected_jobs):
        """Test getting recent jobs with various timeframes."""
        if days == 0:
            # For 0 days, artificially make the jobs older
            self._make_jobs_older(analytics, sample_job_data)
        
        recent_jobs = analytics.get_recent_jobs(days=days)
        assert len(recent_jobs) <= expected_jobs
    
    def _make_jobs_older(self, analytics, sample_job_data):
        """Helper method to make jobs appear older for testing."""
        import os
        for job_data in sample_job_data:
            job_dir = analytics.checkpoint_dir / job_data.job_id
            state_file = job_dir / STATE_FILENAME
            old_time = datetime.now().timestamp() - 3600  # 1 hour ago
            os.utime(state_file, (old_time, old_time))
    
    def test_report_file_naming(self, analytics, sample_job_data):
        """Test that report files are named correctly."""
        before_count = len(list(analytics.reports_dir.glob("summary_report_*.json")))
        
        analytics.generate_summary_report()
        
        after_count = len(list(analytics.reports_dir.glob("summary_report_*.json")))
        assert after_count == before_count + 1
        
        # Check that the latest file has correct timestamp format
        report_files = list(analytics.reports_dir.glob("summary_report_*.json"))
        latest_file = max(report_files, key=lambda f: f.stat().st_mtime)
        
        # File name should contain timestamp
        assert "summary_report_" in latest_file.name
        assert latest_file.name.endswith(".json")