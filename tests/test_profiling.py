"""Tests for the profiling module."""

import time
import pytest
import numpy as np

from crispyx.profiling import Profiler, TimingProfiler, MemoryProfiler


class TestProfilerTiming:
    """Tests for timing functionality of the Profiler class."""
    
    def test_disabled_timing(self):
        """Profiler with timing=False should not record any timings."""
        profiler = Profiler(timing=False)
        profiler.start("section1")
        time.sleep(0.01)
        elapsed = profiler.stop("section1")
        
        assert elapsed == 0.0
        stats = profiler.get_stats()
        assert "timing" not in stats
    
    def test_basic_timing(self):
        """Test basic timing functionality with start/stop."""
        profiler = Profiler(timing=True)
        
        profiler.start("section1")
        time.sleep(0.05)
        elapsed = profiler.stop("section1")
        
        # Should have recorded ~50ms
        assert elapsed > 0.04
        assert elapsed < 0.1
        
        stats = profiler.get_stats()
        assert "timing" in stats
        assert "sections" in stats["timing"]
        assert "section1" in stats["timing"]["sections"]
        assert stats["timing"]["sections"]["section1"]["seconds"] > 0.04
    
    def test_multiple_sections(self):
        """Test timing multiple sections."""
        profiler = Profiler(timing=True)
        
        profiler.start("section1")
        time.sleep(0.02)
        profiler.stop("section1")
        
        profiler.start("section2")
        time.sleep(0.03)
        profiler.stop("section2")
        
        stats = profiler.get_stats()
        assert len(stats["timing"]["sections"]) == 2
        # Section2 should take longer
        assert stats["timing"]["sections"]["section2"]["seconds"] > stats["timing"]["sections"]["section1"]["seconds"]
    
    def test_accumulating_timing(self):
        """Test that repeated start/stop for same label accumulates."""
        profiler = Profiler(timing=True)
        
        profiler.start("loop_section")
        time.sleep(0.01)
        profiler.stop("loop_section")
        
        profiler.start("loop_section")
        time.sleep(0.01)
        profiler.stop("loop_section")
        
        stats = profiler.get_stats()
        # Should have accumulated ~20ms
        assert stats["timing"]["sections"]["loop_section"]["seconds"] > 0.015
    
    def test_total_time(self):
        """Test get_total_time() returns correct elapsed time."""
        profiler = Profiler(timing=True)
        profiler.start("section1")
        time.sleep(0.02)
        profiler.stop("section1")
        
        total = profiler.get_total_time()
        assert total > 0.015


class TestProfilerMemory:
    """Tests for memory functionality of the Profiler class."""
    
    def test_disabled_memory(self):
        """Profiler with memory=False should not take snapshots."""
        profiler = Profiler(memory=False)
        profiler.snapshot("test")
        
        stats = profiler.get_stats()
        assert "memory" not in stats or len(stats.get("memory", {}).get("snapshots", {})) == 0
    
    def test_memory_snapshot_tracemalloc(self):
        """Test memory snapshots with tracemalloc method."""
        import tracemalloc
        
        # Start tracemalloc manually for test
        tracemalloc.start()
        
        profiler = Profiler(memory=True, memory_method="tracemalloc")
        profiler._tracemalloc_start_time = time.perf_counter()
        
        profiler.snapshot("before")
        
        # Allocate some memory
        data = np.zeros((1000, 1000), dtype=np.float64)
        
        profiler.snapshot("after")
        
        stats = profiler.get_stats()
        
        tracemalloc.stop()
        
        assert "memory" in stats
        assert "snapshots" in stats["memory"]
        assert "before" in stats["memory"]["snapshots"]
        assert "after" in stats["memory"]["snapshots"]
        # After should have higher memory than before (we allocated ~8MB)
        assert stats["memory"]["snapshots"]["after"]["current_mb"] >= stats["memory"]["snapshots"]["before"]["current_mb"]
        
        del data
    
    def test_memory_snapshot_rss(self):
        """Test memory snapshots with RSS method."""
        profiler = Profiler(memory=True, memory_method="rss")
        profiler._total_start = time.perf_counter()
        
        profiler.snapshot("test_point")
        
        stats = profiler.get_stats()
        assert "memory" in stats
        assert "test_point" in stats["memory"]["snapshots"]
        # RSS should be non-zero for any running process
        assert stats["memory"]["snapshots"]["test_point"]["current_mb"] > 0


class TestProfilerSampling:
    """Tests for continuous memory sampling functionality."""
    
    def test_sampling_disabled(self):
        """Sampling should not run when sampling=False."""
        profiler = Profiler(sampling=False)
        profiler.start_sampling()
        time.sleep(0.1)
        profiler.stop_sampling()
        
        assert len(profiler._samples) == 0
    
    def test_sampling_enabled(self):
        """Test that memory sampling collects samples."""
        profiler = Profiler(sampling=True, sample_interval=0.02)
        
        profiler.start_sampling()
        time.sleep(0.1)
        profiler.stop_sampling()
        
        # Should have collected approximately 5 samples (100ms / 20ms)
        assert len(profiler._samples) >= 3
        assert len(profiler._samples) <= 10
        
        # Each sample should be (timestamp, memory_mb)
        for timestamp, memory_mb in profiler._samples:
            assert isinstance(timestamp, float)
            assert isinstance(memory_mb, float)
            assert memory_mb > 0


class TestProfilerContextManager:
    """Tests for context manager interface."""
    
    def test_context_manager_timing(self):
        """Test profiler works as context manager with timing."""
        with Profiler(timing=True) as p:
            p.start("inside_context")
            time.sleep(0.02)
            p.stop("inside_context")
        
        stats = p.get_stats()
        assert "timing" in stats
        assert "total" in stats["timing"]["sections"]
        assert "inside_context" in stats["timing"]["sections"]
    
    def test_context_manager_memory(self):
        """Test profiler works as context manager with memory tracking."""
        with Profiler(memory=True) as p:
            p.snapshot("middle")
            data = np.zeros((100, 100))
        
        stats = p.get_stats()
        assert "memory" in stats
        # Context manager should have added 'end' snapshot
        assert "end" in stats["memory"]["snapshots"]


class TestProfilerReport:
    """Tests for report generation."""
    
    def test_empty_report_disabled(self):
        """Report should indicate profiling was disabled."""
        profiler = Profiler(timing=False, memory=False)
        report = profiler.get_report()
        assert "not enabled" in report.lower()
    
    def test_timing_report(self):
        """Test timing report includes section breakdown."""
        profiler = Profiler(timing=True)
        profiler.start("section_a")
        time.sleep(0.01)
        profiler.stop("section_a")
        
        report = profiler.get_report()
        assert "section_a" in report
        assert "Total time" in report or "total" in report.lower()


class TestBackwardCompatibility:
    """Tests for backward-compatible wrapper classes."""
    
    def test_timing_profiler_wrapper(self):
        """TimingProfiler should work as before."""
        timer = TimingProfiler(enabled=True)
        timer.start("section")
        time.sleep(0.01)
        timer.stop("section")
        
        assert "section" in timer.timings
        assert timer.timings["section"] > 0.005
    
    def test_timing_profiler_disabled(self):
        """TimingProfiler with enabled=False should be no-op."""
        timer = TimingProfiler(enabled=False)
        timer.start("section")
        timer.stop("section")
        
        assert len(timer.timings) == 0
    
    def test_memory_profiler_wrapper(self):
        """MemoryProfiler should work as before."""
        import tracemalloc
        tracemalloc.start()
        
        profiler = MemoryProfiler(enabled=True)
        profiler._tracemalloc_start_time = time.perf_counter()
        profiler.snapshot("test")
        
        tracemalloc.stop()
        
        assert "test" in profiler.snapshots


class TestVisualization:
    """Tests for visualization methods (require matplotlib)."""
    
    @pytest.fixture
    def profiler_with_data(self):
        """Create a profiler with some timing data."""
        profiler = Profiler(timing=True, sampling=True, sample_interval=0.01)
        
        profiler.start("total")
        profiler.start_sampling()
        
        profiler.start("section_a")
        time.sleep(0.03)
        profiler.stop("section_a")
        
        profiler.start("section_b")
        time.sleep(0.05)
        profiler.stop("section_b")
        
        profiler.stop_sampling()
        profiler.stop("total")
        
        return profiler
    
    def test_plot_timeline(self, profiler_with_data):
        """Test plot_timeline returns axes object."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for testing
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        ax = profiler_with_data.plot_timeline()
        assert ax is not None
        plt.close()
    
    def test_plot_memory(self, profiler_with_data):
        """Test plot_memory returns axes object when samples exist."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        ax = profiler_with_data.plot_memory()
        assert ax is not None
        plt.close()


class TestIntegration:
    """Integration tests verifying profiler works in realistic scenarios."""
    
    def test_full_profiling_workflow(self):
        """Test a complete profiling workflow with timing, memory, and sampling."""
        with Profiler(timing=True, memory=True, sampling=True, sample_interval=0.01) as profiler:
            profiler.start("data_prep")
            data = np.random.randn(1000, 100)
            profiler.snapshot("after_data_prep")
            profiler.stop("data_prep")
            
            profiler.start("computation")
            result = np.linalg.svd(data, full_matrices=False)
            profiler.snapshot("after_computation")
            profiler.stop("computation")
        
        stats = profiler.get_stats()
        
        # Verify timing stats
        assert "timing" in stats
        assert stats["timing"]["total_seconds"] > 0
        assert "data_prep" in stats["timing"]["sections"]
        assert "computation" in stats["timing"]["sections"]
        
        # Verify memory stats
        assert "memory" in stats
        assert stats["memory"]["peak_mb"] > 0
        assert "after_data_prep" in stats["memory"]["snapshots"]
        assert "after_computation" in stats["memory"]["snapshots"]
        
        # Verify samples were collected
        assert "samples" in stats["memory"]
        assert len(stats["memory"]["samples"]) > 0
        
        # Verify report generation works
        report = profiler.get_report()
        assert len(report) > 100  # Should have substantial content
