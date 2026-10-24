# AI-Assisted Workflow Log

## Project
**Task:** Task 2 — Phase 1: ML Prediction Service  
**AI Coding Assistant:** Claude Code  
**Supporting Assistant:** ChatGPT

## 1. AI Assistance
Used Claude Code to review the project, identify issues, improve validation and error handling, write tests, troubleshoot errors, and update documentation. Proposed changes were reviewed before implementation.

## 2. Suggestions Accepted
- Added tests for invalid inputs, malformed JSON, request-size limits, and model-loading failures.
- Added a 128 KB request-body limit and JSON error responses.
- Improved model-loading error handling after a test exposed an unhandled `KeyError`.
- Configured pytest discovery and updated the README.

## 3. Suggestions Corrected
- Updated exception handling after testing revealed that a corrupted model could raise `KeyError`.
- Fixed pytest discovery using `pythonpath = ["."]` in `pyproject.toml`.
- Repaired the truncated ending of `train.py` and successfully reran model training.

## 4. Suggestions Rejected
- Did not add `IndexError` handling because no demonstrated failure justified it.
- Avoided adding unnecessary defensive code for errors that had not been observed.

## 5. Suggestions Deferred
- Additional Docker hardening, including a non-root user and container health check.
- Gunicorn integration, as it was not required for the current project scope.

These changes were deferred to prioritize core functionality and verification.

## 6. Verification Results
- **Automated tests:** 24 passed, 0 failed.
- **Model accuracy:** 88.07% on 746 test samples.
- **API:** Health, prediction, and invalid-input requests tested successfully.
- **Docker:** Image built and container started successfully.
- **Screenshots:** Training output, test results, API validation, and Docker validation captured.

## 8. Human Review
AI-generated suggestions were reviewed before implementation. Changes were verified using automated tests, model-training output, API responses, and Docker runtime checks. Unverified behavior and remaining limitations were documented rather than treated as completed.