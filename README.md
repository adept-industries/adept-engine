# Adept Engine

The Adept engine is the background analytics and machine-learning service for the Adept platform.

## Responsibilities

This repository owns:

- durable background-job processing;
- GitHub and Jira event normalization;
- pull-request risk feature extraction and inference;
- model artifacts and metadata;
- DORA metric calculations;
- incident and deployment processing;
- alert evaluation;
- offline model-training code.

This repository does not own:

- user authentication;
- public browser-facing API routes;
- React pages;
- Flyway or Alembic schema migrations.

## Technology baseline

- Python 3.12
- uv 0.11.16
- FastAPI
- SQLAlchemy 2
- scikit-learn

## Current status

Phase 0 repository foundation.

The Python project, `pyproject.toml`, `uv.lock`, FastAPI application, and worker will be created during Phase 1.

## Contribution

All changes must be made through a feature branch and pull request after branch protection is enabled.