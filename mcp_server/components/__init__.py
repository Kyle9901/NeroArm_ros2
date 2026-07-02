"""
Atomic component wrappers for the robot task orchestration stack.

Components are thin, deterministic wrappers around RobotBridge, VlmClient, and
VisualServo. They do not perform task-level planning or retry orchestration.
"""

from .base import ComponentResult, ImageFrame
