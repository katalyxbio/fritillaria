pub(crate) mod header;
mod magic_number;
mod reference_sequences;

use std::io::{self, Write};

pub use self::header::write_header;
use self::{
    header::write_aux, magic_number::write_magic_number,
    reference_sequences::write_reference_sequences,
};
use super::num::{write_i32_le, write_u64_le};
use crate::{BinningIndex, Index};

pub(super) fn write_index<W>(writer: &mut W, index: &Index) -> io::Result<()>
where
    W: Write,
{
    write_magic_number(writer)?;

    let min_shift = i32::from(index.min_shift());
    write_i32_le(writer, min_shift)?;

    let depth = i32::from(index.depth());
    write_i32_le(writer, depth)?;

    write_aux(writer, index.header())?;
    write_reference_sequences(writer, index.depth(), index.reference_sequences())?;

    if let Some(n_no_coor) = index.unplaced_unmapped_record_count() {
        write_u64_le(writer, n_no_coor)?;
    }

    Ok(())
}
