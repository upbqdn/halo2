use super::*;
use crate::pasta::vesta;
use group::{Curve, Group, GroupEncoding};

type C = vesta::Affine;
type Plain<'a> = Blake2bRead<&'a [u8], C, Challenge255<C>>;

fn proof(runs: usize) -> Vec<u8> {
    let mut writer = Blake2bWrite::<_, C, Challenge255<C>>::init(Vec::new());
    for _ in 0..runs {
        writer
            .write_point(vesta::Point::generator().to_affine())
            .unwrap();
        writer
            .write_point((vesta::Point::generator().double()).to_affine())
            .unwrap();
        writer.write_scalar(vesta::Scalar::from(7)).unwrap();
    }
    writer.finalize()
}

fn exercise(bytes: &[u8], runs: usize, prefetched: Option<Vec<(usize, Option<C>)>>) {
    let mut plain = Plain::init(bytes);
    let mut fast = match prefetched {
        Some(points) => PrefetchedBlake2bRead::<C>::with_prefetched(bytes, points),
        None => PrefetchedBlake2bRead::<C>::recording(bytes),
    };
    for _ in 0..runs {
        for _ in 0..2 {
            let a = plain.read_point();
            let b = fast.read_point();
            assert_eq!(a.as_ref().ok(), b.as_ref().ok());
            if let Err(a) = a {
                let b = b.unwrap_err();
                assert_eq!(a.kind(), b.kind());
                assert_eq!(a.to_string(), b.to_string());
                return;
            }
        }
        let a = plain.read_scalar();
        let b = fast.read_scalar();
        assert_eq!(a.as_ref().ok(), b.as_ref().ok());
        if let Err(a) = a {
            let b = b.unwrap_err();
            assert_eq!(a.kind(), b.kind());
            assert_eq!(a.to_string(), b.to_string());
            return;
        }
        assert_eq!(
            plain.squeeze_challenge().get_scalar(),
            fast.squeeze_challenge().get_scalar()
        );
    }
}

#[test]
fn prefetched_and_recorded_transcripts_match_scalar_reads_and_errors() {
    let bytes = proof(3);
    for end in 0..=bytes.len() {
        let bytes = &bytes[..end];
        let points = (0..3)
            .flat_map(|run| [run * 96, run * 96 + 32])
            .filter_map(|offset| {
                let encoding = bytes.get(offset..offset + 32)?;
                let encoding: [u8; 32] = encoding.try_into().unwrap();
                Some((offset, C::from_bytes(&encoding).into()))
            })
            .collect();
        exercise(bytes, 3, Some(points));
        exercise(bytes, 3, None);
    }
    for offset in [0, 32, 64, 96] {
        let mut malformed = bytes.clone();
        malformed[offset..offset + 32].fill(255);
        let points = [0, 32, 96, 128, 192, 224]
            .into_iter()
            .map(|offset| {
                let encoding: [u8; 32] = malformed[offset..offset + 32].try_into().unwrap();
                (offset, C::from_bytes(&encoding).into())
            })
            .collect();
        exercise(&malformed, 3, Some(points));
    }
}

#[test]
fn recording_and_shape_caches_remain_bounded() {
    let limit = crate::MAX_CACHED_PROOF_POINT_RUNS;
    for runs in [limit, limit + 1] {
        let bytes = proof(runs);
        let mut recording = PrefetchedBlake2bRead::<C>::recording(&bytes);
        for _ in 0..runs {
            recording.read_point().unwrap();
            recording.read_point().unwrap();
            recording.read_scalar().unwrap();
        }
        let ranges = recording.into_recorded();
        assert_eq!(ranges.is_some(), runs == limit);
        if let Some(ranges) = ranges {
            assert_eq!(
                ranges,
                (0..runs).map(|i| i * 96..i * 96 + 64).collect::<Vec<_>>()
            );
        }
    }
    let layouts = crate::plonk::ProofPointLayouts::default();
    layouts.insert(0, vec![0..32]);
    let in_flight = layouts.get(0).unwrap();
    for circuits in 1..=256 {
        layouts.insert(circuits, vec![32..64]);
    }
    assert!(layouts.get(0).is_none());
    assert_eq!(&*in_flight, &[0..32]);
    layouts.insert(300, (0..=limit).map(|i| i * 32..i * 32 + 32).collect());
    assert!(layouts.get(300).is_none());
}
