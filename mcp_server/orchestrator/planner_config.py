"""
Planning LLM configuration — edit this file to set your model.

Environment variables (fallback if fields are left empty):
  PLANNING_LLM_API_KEY
  PLANNING_LLM_API_URL
  PLANNING_LLM_MODEL
"""

import os

PLANNING_LLM_CONFIG = {
    "api_key": os.environ.get("PLANNING_LLM_API_KEY", "sk-ws-H.RPMHLMD.eNJE.MEUCIBm07fqNtBjq6uFXj5r4XPa_kAGB8mEE0UKuLLZGbc2HAiEAkavdsqSmb1KgE9ZsQUArWKdVpYKSL5Q1QF_PmM_xzsY"),
    "api_url": os.environ.get("PLANNING_LLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
    "model": os.environ.get("PLANNING_LLM_MODEL", "deepseek-v4-flash"),
}