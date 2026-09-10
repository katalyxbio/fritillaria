# Third-party licences

Licences for code this repository redistributes but did not write. The project's
own licence is [`LICENSE`](../LICENSE) at the root — Apache-2.0.

| File | Covers |
|---|---|
| [`noodles-MIT.txt`](noodles-MIT.txt) | [noodles](https://github.com/zaeleus/noodles), © 2018 Michael Macias — vendored into most of `crates/`. See [`VENDORED.md`](../VENDORED.md). |

**Why these are not at the repository root.** GitHub detects a project's licence
by scanning root-level `LICENSE*` files, so a second one there renders as a
second licence tab and reads as though the project were dual-licensed at your
option. It is not: our code is Apache-2.0, the vendored code is MIT, and which
applies depends on the file. The per-crate copies of the MIT notice are
unaffected and are what ships to crates.io.

Nothing here may be deleted or emptied — the MIT licence requires the copyright
notice and the permission notice to travel with the code.

NVIDIA's nvCOMP is not listed, because it is not redistributed: the `nvcomp`
feature dlopens it at runtime from wherever the user installed it, and it is
licensed to that user by NVIDIA directly.
