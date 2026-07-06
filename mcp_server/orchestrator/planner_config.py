"""
Planning LLM configuration — edit this file to set your model.

Environment variables (fallback if fields are left empty):
  PLANNING_LLM_API_KEY
  PLANNING_LLM_API_URL
  PLANNING_LLM_MODEL
"""

import os

PLANNING_LLM_CONFIG = {
    "api_key": os.environ.get("PLANNING_LLM_API_KEY"),
    "api_url": os.environ.get("PLANNING_LLM_API_URL"),
    "model": os.environ.get("PLANNING_LLM_MODEL"),
}