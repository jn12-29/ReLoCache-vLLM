# ReLoCache vLLM Fork

This is an independent Git repository, nested as a submodule of ReLoCache.

The initial Native PIC source was extracted from `sempic_iclr` commit `642fb31bf12c54c74c7b8970067bd4f0da6e3ac1`, subtree `third_party/vllm`, on upstream vLLM commit `08b3e67b669d18d5bbd14e456b8bb4794916e402`. That real upstream commit is retained as the shallow history boundary and tagged `upstream-base`. The next commit consolidates the current ReLoCache execution changes. Use `git diff upstream-base..main` to review all fork changes against upstream.

The `upstream` remote points to `https://github.com/vllm-project/vllm.git`; `origin` points to the private fork at `git@github.com:jn12-29/vllm-relocache.git`.

ReLoCache is a position-independent caching library for efficient inference serving. It owns PIC semantics and core implementations, including PIC-specific inference components, with future support for multiple backends. SemPIC primarily owns training and evaluation semantics. This fork may depend on and call ReLoCache through plugins and extension interfaces; upstream vLLM need not require ReLoCache as an installed dependency.

Integrate ReLoCache's PIC components with the minimum necessary vLLM changes, keeping vLLM-specific details in backend integration. Reuse native KV management, scheduling, cache lookup, and offload mechanisms while preserving performance. Limit core changes to narrow interfaces or execution capabilities suitable for upstream contribution; avoid private implementation coupling and copies of native logic.

This is the target boundary. The fork currently still contains PIC cache identity and placement, request metadata, compilation coordination, pre-RoPE KV, and fused-RoPE attention. Moving these implementations behind extension interfaces remains future work.

Keep source, environment, cache, and experiment directories in the original `sempic_iclr` project read-only. Commit fork changes here before updating the parent repository's submodule reference.
