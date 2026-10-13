#pragma once

struct skippy_session;

// Prepare a pooled session for a non-chat workload. Embedding and rerank
// batches request output for every input row, so no backend sampler may remain
// attached from an earlier chat request.
void skippy_prepare_non_chat_workload(skippy_session * session);
