# Vendored code

Most of this workspace is **derived from [noodles]**, by Michael Macias, and is
used under the MIT licence. The full notice is in `LICENSE-MIT-noodles.txt`, at
the repository root and again in every crate that contains vendored code, so it
travels with the source.

- **Upstream:** <https://github.com/zaeleus/noodles>
- **Commit:** `d1ad3992abfbd93996b5b72f37d493af7dee9e72`
- **Vendored:** 2026-09-09
- **Extent:** all 18 crates, 137,508 lines

Renaming is permitted; stripping attribution is not. The `authors`, `homepage`
and `repository` fields of every vendored crate still point at noodles, and this
file records the derivation. Do not remove either.

## What changed

Mechanical only:

- `noodles-<x>` → `fritillaria-<x>` in package names and dependencies.
- `noodles_<x>` → `fritillaria_<x>` in Rust paths.
- Vendored crates keep upstream's lint set (`missing_docs = "warn"`) rather than
  this workspace's pedantic one. 137k lines we did not write are not going to be
  reformatted to our style, and keeping them diffable against upstream is worth
  more than uniformity.
- `cargo fmt` was run afterwards, touching 75 files. This is not a style change:
  `fritillaria_bam` sorts differently from `noodles_bam`, so rustfmt reorders the
  import blocks that name a renamed crate. Deterministic, and reproduced by
  rerunning the same two steps.

No logic was changed. That is deliberate: it keeps a future rebase onto a newer
noodles a mechanical operation rather than a merge.

**Two hand edits beyond the renames**, both in `fritillaria-bgzf/src/io.rs`. Upstream's
`test_self` and `test_self_multithreaded` write `b"noodles"`, `b"-"`, `b"bgzf"` in three
flushed pieces and assert the reassembled result equals `b"noodles-bgzf"`. The rename matched
the *assertion* — one token — and not the three parts, so the tests failed on data rather than
on behaviour. The writes now spell `fritillaria` to match. What they test did not change.

That is the general hazard: **the rename cannot tell an identifier from a string literal.** It
is harmless in prose and in `add_comment("… by fritillaria-bam")`, and breaks only where a
literal is compared against pieces spelled separately. Two such places existed in 137k lines.

**Nine vendored examples carry an added `#![allow(clippy::pedantic)]`**, seven in
`fritillaria-bam` and two in `fritillaria-bgzf`. Only the *merged* crates need it: they apply
this workspace's lint set, which upstream does not, and `#[allow]` on a module does not reach
`examples/`. Relaxing beat rewriting — the alternative was editing vendored logic to satisfy
lints its author never opted into.

## Verifying the vendoring

Copy upstream to a scratch tree, apply the renames, reformat, and diff against `crates/`.
**Verified 2026-09-09: exactly 14 files differ**, and they are the 14 below.

| Files | Why |
|---|---|
| 4 × `lib.rs` (core, bgzf, bam, bcf) | hand-written to declare both halves |
| `fritillaria-bgzf/src/io.rs` | the two round-trip literals above |
| 9 vendored examples | the added `#![allow(clippy::pedantic)]` |

```bash
SRC=/path/to/noodles-checkout
TMP=$(mktemp -d)
CRATES="bam bcf bed bgzf core cram csi fasta fastq gff gtf htsget refget sam tabix util vcf"
EXPR=""; for c in $CRATES; do
  EXPR="$EXPR -e s/noodles_$c/fritillaria_$c/g -e s/noodles-$c/fritillaria-$c/g"
done

for c in $CRATES; do
  cp -r "$SRC/noodles-$c" "$TMP/fritillaria-$c"
  find "$TMP/fritillaria-$c" -name '*.rs' | xargs sed -i $EXPR
  # From the crate root: rustfmt on a lone file cannot resolve `mod`, and then
  # silently leaves it unformatted — which reads as 39 spurious diffs.
  (cd "$TMP/fritillaria-$c" && rustfmt --edition 2024 --quiet src/lib.rs
   for e in examples/*.rs tests/*.rs benches/*.rs; do
     [ -f "$e" ] && rustfmt --edition 2024 --quiet "$e"
   done)
done

for c in $CRATES; do
  for f in $(cd "$TMP/fritillaria-$c" && find . -name '*.rs'); do
    ours="crates/fritillaria-$c/${f#./}"
    [ -f "$ours" ] && { diff -q "$TMP/fritillaria-$c/$f" "$ours" >/dev/null || echo "$ours"; }
  done
done
```

Run this after any rebase. A file differing unexpectedly is a merge that went wrong. Note the
comment above: formatting file-by-file instead of from the crate root produces a list of
plausible-looking diffs that are entirely artefacts of the check.

## Licensing

| | Licence |
|---|---|
| Vendored crates, unmodified | MIT (© 2018 Michael Macias) |
| `fritillaria`, `fritillaria-cuda` | Apache-2.0 |
| `fritillaria-{core,bgzf,bam,bcf}` | `Apache-2.0 AND MIT` — they contain both |

MIT code may be redistributed inside an Apache-2.0 work; the reverse is not
true, so **nothing here can be contributed back upstream to noodles without
relicensing it**.

## The four merged crates

`core`, `bgzf`, `bam` and `bcf` existed here before vendoring, so each now holds
both halves. The split, per crate:

| Crate | Vendored (MIT) | Ours (Apache-2.0) |
|---|---|---|
| `fritillaria-core` | `position`, `region` | `codec`, `device`, `error`, `virtual_offset` |
| `fritillaria-bgzf` | `io`, `gzi`, `gz`, `deflate`, `async`, `virtual_position` | `block`, `discover`, `cpu`, `read`, `device_read`, `write`, `host_device` |
| `fritillaria-bam` | `io`, `bai`, `fs`, `record`, `record_ref`, `async` | `columnar` |
| `fritillaria-bcf` | `io`, `fs`, `record`, `async` | `columnar` |

`bam` and `bcf` both define a `record` module upstream, and so did we, which is
why ours moved under `columnar`. The name is accurate — that module is the
boundary-scan-and-columns path — and it keeps the *vendored* API at the crate
root, where a caller migrating from noodles expects to find it.

So `fritillaria_bam::Record` is the vendored owning record and
`fritillaria_bam::columnar::Record` is our zero-copy view. Two types, two
purposes, and the drop-in one wins the shorter name.

## Rebasing onto a newer noodles

The changes above are mechanical, so the procedure is:

1. Diff the new upstream against commit `d1ad399`.
2. Apply that diff to the vendored modules listed in the table above.
3. Re-run the two renames.

Nothing in `columnar`, or in the ours column, is touched by that. The one place
a conflict is likely is `lib.rs` in the four merged crates, since those are
hand-written to declare both halves.

[noodles]: https://github.com/zaeleus/noodles
