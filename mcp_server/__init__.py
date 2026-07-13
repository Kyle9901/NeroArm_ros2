"""MCP Server for AGX Robot Arm — LLM + LangGraph orchestration."""


def main():
    """Load the ROS/MCP application only when the entry point is invoked."""
    from .task_server import main as run_server

    return run_server()


__all__ = ["main"]
