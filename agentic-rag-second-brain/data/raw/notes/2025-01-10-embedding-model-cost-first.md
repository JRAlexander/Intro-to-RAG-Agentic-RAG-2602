---
title: Embedding Model Decision: Cost-First Default
date: 2025-01-10
tags:
  - embeddings
  - architecture
  - cost
---

We should standardize on EmbedLite-v1 for now because the projected monthly query volume is high and token costs dominate. Retrieval quality is acceptable in internal tests for broad topical queries.

Decision: Use EmbedLite-v1 as default for all note ingestion pipelines until quality complaints increase.
