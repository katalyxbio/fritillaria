//! Tab-delimited genomic text, scanned into columns.
//!
//! SAM, VCF, BED, GFF and GTF are five formats and one *shape*: newline-
//! delimited records, tab-delimited fields, and a header of lines distinguished
//! by a leading character. This crate is that shape, once, so the device kernel
//! is written once too.
//!
//! # Why one scanner and not five
//!
//! The binary formats did not generalise to one another — BAM's blocked scan is
//! wrong for BCF, and FASTQ's validator is wrong for both. The text formats are
//! the opposite case: their record framing is *identical*, and only the
//! interpretation of the fields differs. Duplicating a delimiter scan five times
//! would be five places for the same bug.
//!
//! # Splitting on tabs is sound, and that is not obvious
//!
//! It rests on two facts, both worth stating because a naive scanner in another
//! ecosystem would need quote tracking:
//!
//! - **A field cannot contain a newline**, by construction: a record *is* a
//!   line.
//! - **A field cannot contain a tab.** All five specs forbid it — GFF3 requires
//!   `%09`, and the others simply disallow it, because a tab would create a
//!   field and no parser could recover.
//!
//! **`"` is not a quote character in SAM, VCF, BED or GFF3.** It is ordinary
//! data — in SAM it is Phred+33 Q1 and appears in quality strings constantly. A
//! first attempt at measuring "tabs inside quotes" reported 1,956 of them in
//! `testdata/reads.sam` purely because it treated `"` as a delimiter. Only GTF
//! quotes attribute values, and only inside the final field, after every tab.
//!
//! So the scan is two independent byte tests and no state machine.
//!
//! # What this deliberately does not do
//!
//! It finds records and fields. It does **not** interpret them: no integer
//! parsing, no CIGAR decoding, no INFO key lookup. Those are per-format and
//! belong in the format crates, and most consumers of a columnar text scan want
//! the spans rather than the values.

pub mod columnar;

pub use columnar::{Dialect, FieldTable, LineKind, RecordBatch, scan_lines};
