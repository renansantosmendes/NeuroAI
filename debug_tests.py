import pytest
import sys

class MyPlugin:
    def pytest_runtest_logreport(self, report):
        if report.failed:
            print(f"FAILED: {report.nodeid}")
            print(report.longrepr)

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_chain.py"], plugins=[MyPlugin()])
