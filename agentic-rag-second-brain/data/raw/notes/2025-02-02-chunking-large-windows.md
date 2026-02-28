---
title: Chunking Strategy v1: Large Windows
date: 2025-02-02
tags:
  - chunking
  - retrieval
---

Initial ingestion uses large chunks (1200 chars, no overlap). The goal is fast indexing and fewer chunks per document.

Decision: keep chunk size large while corpus is still small.
