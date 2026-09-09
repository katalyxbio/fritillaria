use std::io;

use bstr::ByteVec;

use super::RecordBuf;
use crate::{Header, alignment::Record};

impl RecordBuf {
    /// Converts an alignment record to a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use fritillaria_sam::{self as sam, alignment::RecordBuf};
    ///
    /// let header = sam::Header::default();
    /// let record = sam::Record::default();
    ///
    /// let record_buf = RecordBuf::try_from_alignment_record(&header, &record)?;
    ///
    /// assert_eq!(record_buf, RecordBuf::default());
    /// # Ok::<_, std::io::Error>(())
    /// ```
    pub fn try_from_alignment_record<R>(header: &Header, record: &R) -> io::Result<Self>
    where
        R: Record + ?Sized,
    {
        let mut dst = RecordBuf::default();
        dst.try_clone_from_alignment_record(header, record)?;
        Ok(dst)
    }

    /// Clones the given record into this record.
    ///
    /// # Examples
    ///
    /// ```
    /// use fritillaria_sam::{self as sam, alignment::RecordBuf};
    ///
    /// let mut record_buf = RecordBuf::builder().set_name("r1").build();
    ///
    /// let header = sam::Header::default();
    /// let record = sam::Record::default();
    ///
    /// record_buf.try_clone_from_alignment_record(&header, &record)?;
    ///
    /// assert_eq!(record_buf, RecordBuf::default());
    /// # Ok::<_, std::io::Error>(())
    /// ```
    pub fn try_clone_from_alignment_record<R>(
        &mut self,
        header: &Header,
        record: &R,
    ) -> io::Result<()>
    where
        R: Record + ?Sized,
    {
        let name = self.name_mut().take();

        if let Some(src_name) = record.name() {
            *self.name_mut() = if let Some(mut dst_name) = name {
                dst_name.clear();
                dst_name.push_str(src_name);
                Some(dst_name)
            } else {
                Some(src_name.into())
            }
        }

        *self.flags_mut() = record.flags()?;
        *self.reference_sequence_id_mut() = record.reference_sequence_id(header).transpose()?;
        *self.alignment_start_mut() = record.alignment_start().transpose()?;
        *self.mapping_quality_mut() = record.mapping_quality().transpose()?;

        let cigar = self.cigar_mut().as_mut();
        cigar.clear();

        for result in record.cigar().iter() {
            let op = result?;
            cigar.push(op);
        }

        *self.mate_reference_sequence_id_mut() =
            record.mate_reference_sequence_id(header).transpose()?;
        *self.mate_alignment_start_mut() = record.mate_alignment_start().transpose()?;
        *self.template_length_mut() = record.template_length()?;

        let sequence = self.sequence_mut().as_mut();
        sequence.clear();
        sequence.extend(record.sequence().iter());

        let quality_scores = self.quality_scores_mut().as_mut();
        quality_scores.clear();

        for result in record.quality_scores().iter() {
            let score = result?;
            quality_scores.push(score);
        }

        let data = self.data_mut();
        data.clear();

        for result in record.data().iter() {
            let (tag, value) = result?;
            data.insert(tag, value.try_into()?);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use fritillaria_core::Position;

    use super::*;
    use crate::{
        alignment::{
            record::{
                Flags,
                cigar::{Op, op::Kind},
                data::field::Tag,
            },
            record_buf::{Sequence, data::field::Value},
        },
        header::record::value::{Map, map::ReferenceSequence},
    };

    #[test]
    fn test_try_clone_from_alignment_record() -> io::Result<()> {
        let header = Header::builder()
            .add_reference_sequence(
                "sq0",
                Map::<ReferenceSequence>::new(const { NonZero::new(131072).unwrap() }),
            )
            .build();

        let record = RecordBuf::builder()
            .set_flags(Flags::empty())
            .set_reference_sequence_id(0)
            .set_alignment_start(Position::MIN)
            .set_cigar([Op::new(Kind::Match, 4)].into_iter().collect())
            .set_sequence(Sequence::from(b"AAAA"))
            .set_quality_scores([45, 35, 43, 50].into_iter().collect())
            .set_data(
                [(Tag::ALIGNMENT_HIT_COUNT, Value::from(1))]
                    .into_iter()
                    .collect(),
            )
            .build();

        let mut dst = RecordBuf::default();
        dst.try_clone_from_alignment_record(&header, &record)?;

        assert_eq!(dst, record);

        Ok(())
    }
}
