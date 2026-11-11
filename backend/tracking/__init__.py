"""Centralized local/global identity tracking."""

from backend.tracking.cross_camera_tracker import CrossCameraTracker
from backend.tracking.tracking_models import TrackAssignment, TrackingResult

__all__ = ["CrossCameraTracker", "TrackAssignment", "TrackingResult"]
