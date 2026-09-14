# ReLoCache vLLM Fork

This is an independent Git repository, nested as a submodule of ReLoCache.

The initial Native PIC source was extracted from `sempic_iclr` commit `642fb31bf12c54c74c7b8970067bd4f0da6e3ac1`, subtree `third_party/vllm`, on upstream vLLM commit `08b3e67b669d18d5bbd14e456b8bb4794916e402`. That real upstream commit is retained as the shallow history boundary and tagged `upstream-base`. The next commit consolidates the current ReLoCache execution changes. Use `git diff upstream-base..main` to review all fork changes against upstream.

The `upstream` remote points to `https://github.com/vllm-project/vllm.git`; `origin` points to the private fork at `git@github.com:jn12-29/vllm-relocache.git`.

The fork owns execution integration: native cache identity and placement, request metadata, scheduler compilation coordination, offloading, pre-RoPE KV, and fused-RoPE attention. ReLoCache owns the external content and serving interfaces and calls this repository directly.

Keep source, environment, cache, and experiment directories in the original `sempic_iclr` project read-only. Commit fork changes here before updating the parent repository's submodule reference.
