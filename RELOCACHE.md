# ReLoCache vLLM Fork

This is an independent Git repository, nested as a submodule of ReLoCache.

The initial Native PIC source was extracted from `sempic_iclr` commit `642fb31bf12c54c74c7b8970067bd4f0da6e3ac1`, subtree `third_party/vllm`, on upstream vLLM commit `08b3e67b669d18d5bbd14e456b8bb4794916e402`. That real upstream commit is retained as the shallow history boundary and tagged `upstream-base`. The next commit consolidates the current ReLoCache execution changes. Use `git diff upstream-base..main` to review all fork changes against upstream.

The `upstream` remote points to `https://github.com/vllm-project/vllm.git`; `origin` points to the private fork at `git@github.com:jn12-29/vllm-relocache.git`.

ReLoCache targets efficient PIC-based inference serving and implements PIC semantics and core components, including PIC-specific inference components, with future support for multiple backends. SemPIC primarily owns training and evaluation semantics. This fork may depend on and call ReLoCache through plugins and extension interfaces; upstream vLLM need not require ReLoCache as an installed dependency.

Integrate ReLoCache's PIC components with the minimum necessary vLLM changes, keeping vLLM-specific details in backend integration. Reuse native KV management, scheduling, cache lookup, and offload mechanisms while preserving performance. Limit core changes to narrow interfaces or execution capabilities suitable for upstream contribution; avoid private implementation coupling and copies of native logic.

Content categories are **Reusable Content** and **Dynamic Content**. Reusable Content is independently compiled into PIC KV for reuse at different logical positions. Dynamic Content executes in the current Reader context and retains native prefix-cache reuse; its code identifiers use `dynamic_content`.

PIC sources, identities, compilation coordination, logical placement, range traversal, key rotation and their tests live in ReLoCache. Its vLLM adapter specializes Scheduler, Request and ModelState and registers query-only RoPE with the existing CustomOp registry. The fork has no ReLoCache imports or content-protocol names.

The cumulative changes against `upstream-base` provide opaque request metadata with typed tensors, custom request construction and cache admission, internal requests, worker metadata and ModelState selection, logical position mapping and fused attention transforms, scoped offload work tracking/transfer statistics, and request error/stream metadata propagation. Rendering callbacks and explicit all-replica utility results keep service policy outside native code. Residency inspection composes existing tier lookup/lifecycle methods in ReLoCache. Native KV ownership, queues, cache lookup, tiered transfers and attention tiling/softmax remain native. The parent repository documents each retained change in `docs/vllm-patch.md`.

Recompute methods such as EPIC and CacheBlend, and compression methods, are deferred rather than removed. Protocol and validation details are maintained in the parent repository's `docs/integration.md` and `docs/decoupling-validation.md`.

Keep source, environment, cache, and experiment directories in the original `sempic_iclr` project read-only. Commit fork changes here before updating the parent repository's submodule reference.
