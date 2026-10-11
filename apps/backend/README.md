---
title: VisionScale Backend
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

FastAPI inference service for VisionScale, packaged as a Docker Space.

- Health: `/health`
- Version: `/version`
- Metrics: `/metrics`
- Port: `${PORT}` (exposed by Space, set to 7860 in Dockerfile)
