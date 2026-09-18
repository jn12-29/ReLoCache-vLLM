# ReLoCache vLLM Fork

This is an independent Git repository, nested as a submodule of ReLoCache.

The initial Native PIC source was extracted from `sempic_iclr` commit `642fb31bf12c54c74c7b8970067bd4f0da6e3ac1`, subtree `third_party/vllm`, on upstream vLLM commit `08b3e67b669d18d5bbd14e456b8bb4794916e402`. That real upstream commit is retained as the shallow history boundary and tagged `upstream-base`. The next commit consolidates the current ReLoCache execution changes. Use `git diff upstream-base..main` to review all fork changes against upstream.

The `upstream` remote points to `https://github.com/vllm-project/vllm.git`; `origin` points to the private fork at `git@github.com:jn12-29/vllm-relocache.git`.

ReLoCache targets efficient PIC-based inference serving and implements PIC semantics and core components, including PIC-specific inference components, with future support for multiple backends. SemPIC primarily owns training and evaluation semantics. This fork may depend on and call ReLoCache through plugins and extension interfaces; upstream vLLM need not require ReLoCache as an installed dependency.

Integrate ReLoCache's PIC components with the minimum necessary vLLM changes, keeping vLLM-specific details in backend integration. Reuse native KV management, scheduling, cache lookup, and offload mechanisms while preserving performance. Limit core changes to narrow interfaces or execution capabilities suitable for upstream contribution; avoid private implementation coupling and copies of native logic.

Content categories are **Reusable Content** and **Dynamic Content**. Reusable Content is independently compiled into PIC KV for reuse at different logical positions. Dynamic Content executes in the current Reader context and retains native prefix-cache reuse; its code identifiers use `dynamic_content`.

Source snapshots and logical Reader input assembly now live in ReLoCache core modules. Compilation request construction and PIC page identity have moved to `relocache.backends.vllm.compiler`, called only on PIC execution paths. Native completion rendering exposes an optional input callback alongside the chat callback, preserving native generation and streaming handlers.

The fork still contains request metadata, scheduling coordination, physical/logical storage placement, pre-RoPE KV, and fused-RoPE attention. The current priority is to decouple these PIC components through extension interfaces, completing required functionality along the same paths. Recompute methods such as EPIC and CacheBlend, and compression methods, are deferred. The parent repository's `docs/roadmap.md` tracks the work.

Keep source, environment, cache, and experiment directories in the original `sempic_iclr` project read-only. Commit fork changes here before updating the parent repository's submodule reference.
