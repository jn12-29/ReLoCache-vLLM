# ReLoCache-vLLM

This fork provides native execution extensions for
[ReLoCache](https://github.com/jn12-29/ReLoCache), a PIC-based reusable-content
serving package. It is maintained separately from upstream vLLM.

## Install and run

Use the parent repository's recursive checkout and locked environment:

```bash
git clone --recurse-submodules https://github.com/jn12-29/ReLoCache.git
cd ReLoCache
uv sync --python 3.12 --locked --extra server --extra webchat
uv run --no-sync relocache serve \
  --config examples/quickstart/deployment.json \
  --cuda-visible-devices 0 --host 127.0.0.1 --port 8000
```

Select a GPU allocated to you. The parent pins this fork and configures the
matching precompiled vLLM extensions. Installing upstream vLLM from PyPI does
not provide these extensions. The fork's Python package is still named `vllm`.

## Integration boundary

The upstream base is `08b3e67b669d18d5bbd14e456b8bb4794916e402`, tagged
`upstream-base`. Compare the full patch from this directory:

```bash
git diff upstream-base..HEAD
```

ReLoCache owns content identity, compilation policy, logical position mapping,
pre-RoPE key transforms, and backend adapters. This fork contains no imports of
ReLoCache and does not require it for ordinary vLLM use. It retains native KV
ownership, scheduling, offloading, and attention loops, with extensions for:

- Request construction, required-cache admission, and internal execution.
- Renderer and ModelState selection.
- Fused attention traversal, logical masking, and key transforms.
- Request-scoped offload completion and transfer statistics.
- Request errors and streaming KV metadata.
- Shared offload memory cleanup.

The extensions are not merged upstream. PIC integration and regression tests
live in the parent repository. See the parent's
[patch boundary](https://github.com/jn12-29/ReLoCache/blob/main/docs/vllm-patch.md),
[validation scope](https://github.com/jn12-29/ReLoCache/blob/main/docs/decoupling-validation.md),
and [contribution guide](https://github.com/jn12-29/ReLoCache/blob/main/CONTRIBUTING.md).
Qwen3 and the documented configurations have execution checks; these do not
establish support for every upstream model or backend.

## Issues, contributions, and license

Report ReLoCache integration issues in the
[ReLoCache issue tracker](https://github.com/jn12-29/ReLoCache/issues), including
both repository revisions and a minimal reproduction. General upstream vLLM
issues should follow the upstream project's reporting guidelines.

Follow this repository's `AGENTS.md` and upstream contribution policies when
editing vLLM. Commit changes here before updating the parent's submodule pin.
Inherited upstream automation is scoped to `vllm-project`; it is not a CI result
for this fork. Integration checks are maintained by ReLoCache.

The fork retains the upstream [Apache-2.0 license](LICENSE) and source
attribution. Modified execution files identify the ReLoCache modifications.
