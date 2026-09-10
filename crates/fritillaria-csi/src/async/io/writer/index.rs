mod header;
mod magic_number;
mod reference_sequences;

use tokio::io::{self, AsyncWrite, AsyncWriteExt};

use self::{
    header::write_aux, magic_number::write_magic_number,
    reference_sequences::write_reference_sequences,
};
use crate::{BinningIndex, Index};

pub(super) async fn write_index<W>(writer: &mut W, index: &Index) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_magic_number(writer).await?;

    let min_shift = i32::from(index.min_shift());
    writer.write_i32_le(min_shift).await?;

    let depth = i32::from(index.depth());
    writer.write_i32_le(depth).await?;

    write_aux(writer, index.header()).await?;
    write_reference_sequences(writer, index.depth(), index.reference_sequences()).await?;

    if let Some(n) = index.unplaced_unmapped_record_count() {
        writer.write_u64_le(n).await?;
    }

    Ok(())
}
