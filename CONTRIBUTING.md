# Contributing to Smedjan

Thanks for your interest. Smedjan is a pure-Rust LLM engine with no Python and a deliberately tiny dependency tree — contributions should keep it that way.

## Ground rules

- **No new heavyweight dependencies.** The point of the project is owning the stack. If you need a new crate, open an issue first and explain why it can't be a few hundred lines of in-tree Rust.
- **GPU correctness is verified on hardware.** The test suite dispatches real Metal/CUDA kernels; there is no CPU fallback for most ops. Run the tests on the backend you changed before sending a PR.
- **New kernels need a gradient check.** Any new differentiable op must come with a finite-difference or analytic-equivalence gradient test. Many subtle bugs (buffer aliasing, missing zero-init on scatter-add) only surface under a real gradient check, not a forward-value test.

## Building

```bash
cargo build --release                                          # Metal (macOS / Apple Silicon)
cargo build --release --no-default-features --features cuda    # CUDA (NVIDIA)
```

## Testing

```bash
cargo test --release                                  # Metal
cargo test --release -- --include-ignored --test-threads=1   # serial GPU tests
cargo test --release --no-default-features --features cuda    # CUDA
```

A green Metal suite does not imply CUDA passes — verify both backends separately when you touch shared logic.

## Pull requests

1. Keep changes focused; one logical change per PR.
2. Run `cargo fmt` and `cargo clippy` and fix warnings in the files you touched.
3. Include tests for new behavior; for kernels, include a gradient check.
4. Describe what you changed and how you verified it (which backend, which machine).

## Reporting issues

Open an issue with: the command you ran, the backend (Metal/CUDA) and machine, the model size/config, and the full output. For training-divergence reports, include the loss curve and the sequence length — many issues are sequence-length-specific.

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
