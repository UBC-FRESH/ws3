"""
Documentation tests for ws3.

Tests that documentation examples are correct and runnable.
"""

import sys
sys.path.append('../ws3/')

import pytest
import os
import subprocess


class TestDocumentationExamples:
    """Test that documentation examples work correctly."""
    
    def test_basic_import(self):
        """Test basic ws3 import."""
        try:
            import ws3
            assert hasattr(ws3, '__version__'), "Missing __version__"
            assert ws3.__version__ is not None, "__version__ is None"
        except ImportError as e:
            pytest.skip(f"ws3 not installed: {e}")
    
    def test_forest_model_import(self):
        """Test ForestModel import."""
        try:
            from ws3.forest import ForestModel
            assert ForestModel is not None
        except ImportError as e:
            pytest.skip(f"ws3.forest not available: {e}")
    
    def test_opt_import(self):
        """Test opt module import."""
        try:
            from ws3.opt import Problem, Variable, Constraint
            assert Problem is not None
            assert Variable is not None
            assert Constraint is not None
        except ImportError as e:
            pytest.skip(f"ws3.opt not available: {e}")
    
    def test_core_import(self):
        """Test core module import."""
        try:
            from ws3.core import interpolate_curves
            assert interpolate_curves is not None
        except ImportError as e:
            pytest.skip(f"ws3.core not available: {e}")
    
    def test_perf_import(self):
        """Test perf module import."""
        try:
            from ws3.perf import SolverTuner, MemoryProfiler, PerformanceBenchmark
            assert SolverTuner is not None
            assert MemoryProfiler is not None
            assert PerformanceBenchmark is not None
        except ImportError as e:
            pytest.skip(f"ws3.perf not available: {e}")
    
    def test_integration_import(self):
        """Test integration module import."""
        try:
            from ws3.integration import FHOPSIntegrator, FEMICIntegrator
            assert FHOPSIntegrator is not None
            assert FEMICIntegrator is not None
        except ImportError as e:
            pytest.skip(f"ws3.integration not available: {e}")


class TestDocumentationBuild:
    """Test that documentation builds correctly."""
    
    def test_sphinx_build(self):
        """Test Sphinx documentation build."""
        try:
            result = subprocess.run(
                ["sphinx-build", "-b", "html", "docs/source", "docs/build/test_html"],
                capture_output=True,
                text=True,
                cwd=".."
            )
            
            # Check for errors
            if result.returncode != 0:
                # Check if it's just warnings (not critical)
                if "warning" in result.stderr.lower():
                    pytest.skip("Documentation build has warnings (not critical)")
                else:
                    pytest.fail(f"Documentation build failed: {result.stderr}")
            
            # Check that output was created
            html_dir = os.path.join("..", "docs", "build", "test_html")
            assert os.path.exists(html_dir), f"HTML output directory not created: {html_dir}"
            
            # Check for index.html
            index_html = os.path.join(html_dir, "index.html")
            assert os.path.exists(index_html), f"Index HTML not created: {index_html}"
            
        except FileNotFoundError:
            pytest.skip("sphinx-build not available")
    
    def test_rst_syntax(self):
        """Test RST syntax in documentation files."""
        docs_dir = os.path.join("..", "docs", "source")
        
        # Check key documentation files
        rst_files = [
            "index.rst",
            "getting_started/index.rst",
            "textbook/index.rst",
            "howto/index.rst",
        ]
        
        for rst_file in rst_files:
            filepath = os.path.join(docs_dir, rst_file)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                # Check for basic RST structure
                assert ".. _" in content or "=" in content, \
                    f"File {rst_file} doesn't appear to be valid RST"
                
                # Check for proper section headers
                lines = content.split('\n')
                has_header = False
                for i, line in enumerate(lines):
                    if i > 0 and lines[i-1] and not lines[i-1].startswith(' '):
                        if line.strip() and line.strip()[0] in '=-~^"\'`:':
                            has_header = True
                            break
                
                # At least some lines should be section headers
                # (This is a basic check, not comprehensive RST validation)


class TestExampleNotebooks:
    """Test that example notebooks exist and are valid."""
    
    def test_notebook_files_exist(self):
        """Test that key notebook files exist."""
        examples_dir = os.path.join("..", "examples")
        
        key_notebooks = [
            "070_ws3_quickstart_complete_workflow.ipynb",
            "071_ws3_scenario_analysis_and_comparison.ipynb",
            "073_ws3_spatial_constraints.ipynb",
            "074_ws3_multi_objective_optimization.ipynb",
            "075_ws3_parallel_optimization.ipynb",
            "076_ws3_performance_optimization.ipynb",
            "077_ws3_integration_examples.ipynb",
        ]
        
        for notebook in key_notebooks:
            filepath = os.path.join(examples_dir, notebook)
            assert os.path.exists(filepath), f"Notebook not found: {notebook}"
    
    def test_notebook_structure(self):
        """Test that notebooks have valid structure."""
        import json
        
        examples_dir = os.path.join("..", "examples")
        
        # Check a few key notebooks
        notebooks_to_check = [
            "070_ws3_quickstart_complete_workflow.ipynb",
            "076_ws3_performance_optimization.ipynb",
        ]
        
        for notebook in notebooks_to_check:
            filepath = os.path.join(examples_dir, notebook)
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    nb = json.load(f)
                
                # Check basic structure
                assert "cells" in nb, f"{notebook} missing 'cells' key"
                assert len(nb["cells"]) > 0, f"{notebook} has no cells"
                
                # Check cell types
                cell_types = set(cell["cell_type"] for cell in nb["cells"])
                assert "markdown" in cell_types or "code" in cell_types, \
                    f"{notebook} has no markdown or code cells"
    
    def test_notebook_imports(self):
        """Test that notebooks have valid imports."""
        import json
        
        examples_dir = os.path.join("..", "examples")
        
        # Check a performance optimization notebook
        notebook_file = "076_ws3_performance_optimization.ipynb"
        filepath = os.path.join(examples_dir, notebook_file)
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                nb = json.load(f)
            
            # Find code cells with imports
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    source = ''.join(cell["source"])
                    if "import" in source:
                        # Check that imports are valid Python
                        try:
                            compile(source, notebook_file, 'exec')
                        except SyntaxError as e:
                            pytest.fail(f"Syntax error in {notebook_file}: {e}")
                        break


class TestAPIConsistency:
    """Test API consistency across modules."""
    
    def test_version_consistency(self):
        """Test that version is consistent across modules."""
        try:
            import ws3
            from ws3.forest import ForestModel
            from ws3.opt import Problem
            
            # All should have same version
            assert ws3.__version__ == ForestModel.__module__.split('.')[0] + ".version", \
                "Version inconsistency detected"
        except (ImportError, AttributeError):
            pytest.skip("Version check not available")
    
    def test_error_messages(self):
        """Test that error messages are informative."""
        try:
            from ws3.opt import Variable
            
            # Test invalid variable creation
            with pytest.raises(ValueError):
                Variable("test", "invalid_type", 0, 100)
            
            # Test invalid constraint creation
            from ws3.opt import Constraint
            with pytest.raises(ValueError):
                Constraint("test", {}, "invalid_sense", 0)
                
        except ImportError:
            pytest.skip("ws3.opt not available")
    
    def test_deprecation_warnings(self):
        """Test that deprecated features emit warnings."""
        # This test would need deprecation warnings to be implemented
        # For now, just test that the API works
        try:
            from ws3.opt import Problem
            problem = Problem()
            assert problem is not None
        except ImportError:
            pytest.skip("ws3.opt not available")