use ff::Field;
use group::{ff::PrimeField, Curve, Group};
use pasta_curves::{EqAffine, Fp};
use rand::rngs::SysRng;
use rand_core::UnwrapErr;

use super::super::InstanceCommitments;
use super::*;
use crate::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    pasta::{pallas, vesta},
    plonk::{
        create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column,
        ConstraintSystem, Error, Instance, ProvingKey, SingleVerifier, VerifyingKey,
    },
    poly::commitment::{table_pays_off, Params},
    transcript::{
        Blake2bRead, Blake2bWrite, Challenge255, EncodedChallenge, Transcript, TranscriptWrite,
    },
};

#[derive(Clone)]
struct TwoInstanceColumns;

impl Circuit<Fp> for TwoInstanceColumns {
    type Config = ();
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        meta.instance_column();
        meta.instance_column();
    }

    fn synthesize(&self, _: (), _: impl Layouter<Fp>) -> Result<(), Error> {
        Ok(())
    }
}

/// The challenge after absorbing `vk` and the commitments, as a proof's
/// transcript does before reading anything from the proof.
fn challenge(vk: &VerifyingKey<EqAffine>, commitments: &InstanceCommitments<EqAffine>) -> Fp {
    let mut transcript = Blake2bWrite::<_, _, Challenge255<EqAffine>>::init(vec![]);
    vk.hash_into(&mut transcript).unwrap();
    for commitment in commitments.0.iter().flatten() {
        transcript.common_point(*commitment).unwrap();
    }
    transcript.squeeze_challenge().get_scalar()
}

#[test]
fn shared_commitments_match_each_proof_alone() {
    let params = Params::<EqAffine>::new(7);
    let vk = keygen_vk(&params, &TwoInstanceColumns).unwrap();
    let mut proofs = vec![
        vec![
            vec![vec![], vec![Fp::from(3); 10]],
            vec![vec![Fp::from(5); 65], vec![Fp::from(7); 64]],
        ],
        vec![],
        vec![vec![vec![Fp::from(11)], vec![Fp::ZERO; 17]]],
    ];
    // The proofs above hold five columns of at most 64 rows, and each proof
    // added below two of nine and three rows: enough of them that the shared
    // commitment takes the table, while each proof alone does not.
    let extra = (1..)
        .find(|&proofs| table_pays_off(64, 5 + 2 * proofs))
        .unwrap();
    assert!(!table_pays_off(64, 3) && !table_pays_off(9, 2));
    proofs.extend((0..extra).map(|seed| {
        let seed = Fp::from(u64::try_from(seed).unwrap());
        vec![vec![vec![-seed; 9], vec![seed.square(); 3]]]
    }));

    let shared = InstanceCommitments::for_proofs(&params, &vk, &proofs).unwrap();
    assert_eq!(shared.len(), proofs.len());
    // The shared commitment cached a table on `params`, and clones share it,
    // so commit each proof alone on fresh parameters to take the generic path.
    let fresh = Params::<EqAffine>::new(7);
    for (proof, shared) in proofs.iter().zip(&shared) {
        let alone =
            InstanceCommitments::for_proofs(&fresh, &vk, core::slice::from_ref(proof)).unwrap();
        assert_eq!(shared.0, alone[0].0);
        assert_eq!(
            shared.0.iter().map(Vec::len).collect::<Vec<_>>(),
            proof.iter().map(Vec::len).collect::<Vec<_>>(),
        );
        assert_eq!(challenge(&vk, shared), challenge(&vk, &alone[0]));
    }

    let no_proofs: [Vec<Vec<Vec<Fp>>>; 0] = [];
    assert!(InstanceCommitments::for_proofs(&params, &vk, &no_proofs)
        .unwrap()
        .is_empty());
}

#[test]
fn instance_shapes_are_checked_before_committing() {
    let params = Params::<EqAffine>::new(7);
    let vk = keygen_vk(&params, &TwoInstanceColumns).unwrap();
    let max_len = usize::try_from(params.n).unwrap() - (vk.cs.blinding_factors() + 1);
    for (circuit, expected_too_large) in [
        (vec![vec![]], false),
        (vec![vec![], vec![], vec![]], false),
        (vec![vec![Fp::ZERO; max_len + 1], vec![]], true),
    ] {
        // A valid proof first: one invalid proof rejects the whole batch.
        let proofs = [vec![vec![vec![], vec![]]], vec![circuit]];
        let result = InstanceCommitments::for_proofs(&params, &vk, &proofs);
        if expected_too_large {
            assert!(matches!(result, Err(Error::InstanceTooLarge)));
        } else {
            assert!(matches!(result, Err(Error::InvalidInstances)));
        }
    }
    let maximal = [vec![vec![vec![Fp::ZERO; max_len], vec![]]]];
    assert!(InstanceCommitments::for_proofs(&params, &vk, &maximal).is_ok());
}

#[test]
fn repeated_anchor_batches_bind_each_instances_anchor() {
    let params = Params::<EqAffine>::new(5);
    let vk = keygen_vk(&params, &TwoInstanceColumns).unwrap();
    let pk = keygen_pk(&params, vk, &TwoInstanceColumns).unwrap();
    let proofs = (0u64..8)
        .map(|index| {
            let columns = (0..2)
                .map(|column| {
                    let mut values = vec![Fp::from(index + column + 2); 7];
                    values[0] = Fp::from(index / 2);
                    values.extend([Fp::ONE, Fp::ZERO, Fp::ONE]);
                    values
                })
                .collect::<Vec<_>>();
            let slices = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
            let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
            create_proof(
                &params,
                &pk,
                &[TwoInstanceColumns],
                &[&slices],
                UnwrapErr(SysRng),
                &mut transcript,
            )
            .unwrap();
            (vec![columns], transcript.finalize())
        })
        .collect::<Vec<_>>();
    for alteration in 0..5 {
        let mut batch = super::BatchVerifier::new();
        for (index, (instances, proof)) in proofs.iter().enumerate() {
            let mut instances = instances.clone();
            let mut proof = proof.clone();
            if index == 3 {
                match alteration {
                    0 => {}
                    // Change only one column of a repeated anchor group.
                    1 => instances[0][1][0] += Fp::ONE,
                    // Move both columns to another existing anchor group.
                    2 => {
                        instances[0][0][0] = Fp::ZERO;
                        instances[0][1][0] = Fp::ZERO;
                    }
                    // Exercise the generic fallback after sharing would apply.
                    3 => instances[0][1][7] = Fp::from(2),
                    _ => {
                        proof.pop();
                    }
                }
            }
            batch.add_proof(instances, proof);
        }
        assert_eq!(batch.finalize(&params, pk.get_vk()), alteration == 0);
    }
}

type C = vesta::Affine;
type F = vesta::Scalar;

#[derive(Clone)]
struct TestCircuit(Value<F>);

impl Circuit<F> for TestCircuit {
    type Config = (Column<Advice>, Column<Instance>);
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self(Value::unknown())
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let advice = meta.advice_column();
        let instance = meta.instance_column();
        meta.enable_equality(advice);
        meta.enable_equality(instance);
        (advice, instance)
    }

    fn synthesize(
        &self,
        (advice, instance): Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let cell = layouter.assign_region(
            || "public value",
            |mut region| region.assign_advice(|| "value", advice, 0, || self.0),
        )?;
        layouter.constrain_instance(cell.cell(), instance, 0)
    }
}

fn make_proof(params: &Params<C>, pk: &ProvingKey<C>, circuits: usize) -> Vec<u8> {
    let values = [F::from(7)];
    let columns = [values.as_slice()];
    let instances = vec![columns.as_slice(); circuits];
    let circuits = vec![TestCircuit(Value::known(values[0])); circuits];
    let mut transcript = Blake2bWrite::<_, C, Challenge255<C>>::init(Vec::new());
    create_proof(
        params,
        pk,
        &circuits,
        &instances,
        UnwrapErr(SysRng),
        &mut transcript,
    )
    .unwrap();
    transcript.finalize()
}

fn instances(circuits: usize, value: u64) -> Vec<Vec<Vec<F>>> {
    vec![vec![vec![F::from(value)]]; circuits]
}

fn prefetch_differential<C: CurveAffine>()
where
    C::Scalar: FromUniformBytes<64>,
{
    let mut writer = Blake2bWrite::<_, C, Challenge255<C>>::init(Vec::new());
    let point = C::Curve::generator().to_affine();
    writer.write_point(point).unwrap();
    writer.write_scalar(C::Scalar::from(11)).unwrap();
    writer.write_point(point).unwrap();
    let bytes = writer.finalize();
    let len = C::Repr::default().as_ref().len();
    let scalar_len = <C::Scalar as group::ff::PrimeField>::Repr::default()
        .as_ref()
        .len();
    let second = len + scalar_len;
    let ranges = [0..len, second..second + len];
    for end in 0..=bytes.len() {
        let truncated = &bytes[..end];
        let prefetched = prefetch_points::<C>(truncated, &ranges);
        let expected: Vec<_> = [0, second]
            .into_iter()
            .filter(|offset| offset + len <= end)
            .map(|offset| (offset, Some(point)))
            .collect();
        assert_eq!(prefetched, expected);
    }
    let mut malformed = bytes.clone();
    malformed[..len].fill(255);
    assert_eq!(
        prefetch_points::<C>(&malformed, &ranges),
        vec![(0, None), (second, Some(point))]
    );
    malformed[..len].copy_from_slice(C::identity().to_bytes().as_ref());
    assert_eq!(
        prefetch_points::<C>(&malformed, &ranges)[0],
        (0, Some(C::identity()))
    );
}

#[test]
fn batched_decodings_match_individual_decodings_and_skip_partial_points() {
    prefetch_differential::<pallas::Affine>();
    prefetch_differential::<vesta::Affine>();
}

#[test]
fn only_successful_batches_install_layouts_and_clones_share_them() {
    let params = Params::<C>::new(4);
    let circuit = TestCircuit(Value::unknown());
    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk.clone(), &circuit).unwrap();
    let one = make_proof(&params, &pk, 1);
    let two = make_proof(&params, &pk, 2);

    // This proof parses completely, but its changed public input fails the MSM.
    let wrong_values = [F::from(8)];
    let wrong_columns = [wrong_values.as_slice()];
    let mut plain = Blake2bRead::<_, C, Challenge255<C>>::init(one.as_slice());
    let parsed = verify_proof(
        &params,
        &vk,
        BatchStrategy::new(&params, F::ONE),
        &[wrong_columns.as_slice()],
        &mut plain,
    )
    .unwrap();
    assert!(!parsed.eval());
    let mut invalid = BatchVerifier::new();
    invalid.add_proof(instances(1, 8), one.clone());
    assert!(!invalid.finalize(&params, &vk));
    assert!(vk.proof_point_layouts().get(1).is_none());

    // Even the valid member of a failing batch must not publish a layout.
    let mut mixed = BatchVerifier::new();
    mixed.add_proof(instances(1, 7), one.clone());
    mixed.add_proof(instances(2, 8), two.clone());
    assert!(!mixed.finalize(&params, &vk));
    assert!(vk.proof_point_layouts().get(1).is_none());
    assert!(vk.proof_point_layouts().get(2).is_none());

    let mut truncated = BatchVerifier::new();
    truncated.add_proof(instances(1, 7), one[..one.len() - 1].to_vec());
    assert!(!truncated.finalize(&params, &vk));
    assert!(vk.proof_point_layouts().get(1).is_none());

    let clone = vk.clone();
    let pinned = format!("{:?}", vk.pinned());
    let mut cold = BatchVerifier::new();
    cold.add_proof(instances(1, 7), one.clone());
    cold.add_proof(instances(2, 7), two.clone());
    assert!(cold.finalize(&params, &vk));
    for circuits in [1, 2] {
        let ranges = vk.proof_point_layouts().get(circuits).unwrap();
        assert!(ranges.len() <= crate::MAX_CACHED_PROOF_POINT_RUNS);
        assert!(std::sync::Arc::ptr_eq(
            &ranges,
            &clone.proof_point_layouts().get(circuits).unwrap()
        ));
    }
    assert_eq!(pinned, format!("{:?}", clone.pinned()));
    let mut warm = BatchVerifier::new();
    warm.add_proof(instances(2, 7), two);
    warm.add_proof(instances(1, 7), one.clone());
    assert!(warm.finalize(&params, &clone));

    // Differential full-verifier errors with a learned layout, including every
    // truncation and well-formed proofs rejected only by the final MSM check.
    let ranges = vk.proof_point_layouts().get(1).unwrap();
    let values = [F::from(7)];
    let columns = [values.as_slice()];
    let inputs = [columns.as_slice()];
    let compare = |bytes: &[u8]| {
        let mut plain = Blake2bRead::<_, C, Challenge255<C>>::init(bytes);
        let mut fast =
            PrefetchedBlake2bRead::with_prefetched(bytes, prefetch_points::<C>(bytes, &ranges));
        let a = verify_proof(
            &params,
            &vk,
            SingleVerifier::new(&params),
            &inputs,
            &mut plain,
        );
        let b = verify_proof(
            &params,
            &vk,
            SingleVerifier::new(&params),
            &inputs,
            &mut fast,
        );
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    };
    for end in 0..=one.len() {
        compare(&one[..end]);
    }
    for range in ranges.iter() {
        let mut malformed = one.clone();
        malformed[range.start..range.start + 32].fill(255);
        compare(&malformed);
    }
    let mut changed_scalar = one;
    let scalar_start = ranges[0].end;
    changed_scalar[scalar_start..scalar_start + 32].copy_from_slice(F::from(23).to_repr().as_ref());
    compare(&changed_scalar);
}

#[test]
fn known_key_final_msm_matches_old_on_proofs_and_errors() {
    let params = Params::<C>::new(4);
    let circuit = TestCircuit(Value::unknown());
    let vk = keygen_vk(&params, &circuit).unwrap();
    let pk = keygen_pk(&params, vk.clone(), &circuit).unwrap();
    let known = std::sync::Arc::new(KnownTerms::new(
        vk.fixed_commitments
            .iter()
            .chain(vk.permutation.commitments())
            .copied(),
    ));
    let proofs = (0..8)
        .map(|_| make_proof(&params, &pk, 1))
        .collect::<Vec<_>>();

    for count in 1..=proofs.len() {
        for altered in [0, 1, 2, 3] {
            let mut old = params.empty_msm();
            let mut optimized = params.empty_msm();
            for (index, bytes) in proofs[..count].iter().enumerate() {
                let mut bytes = bytes.clone();
                let instance = instances(
                    1,
                    if altered == 1 && index + 1 == count {
                        8
                    } else {
                        7
                    },
                );
                if index + 1 == count {
                    if altered == 2 {
                        bytes.pop();
                    } else if altered == 3 {
                        bytes[..32].fill(255);
                    }
                }
                let mut commitments =
                    InstanceCommitments::for_proofs(&params, &vk, &[&instance, &instance])
                        .unwrap()
                        .into_iter();
                let rho = F::from(index as u64 + 1);
                let mut old_transcript = PrefetchedBlake2bRead::recording(bytes.as_slice());
                let mut new_transcript = PrefetchedBlake2bRead::recording(bytes.as_slice());
                let old_proof = verify_proof_with_instance_commitments(
                    &params,
                    &vk,
                    BatchStrategy::new(&params, rho),
                    commitments.next().unwrap(),
                    &mut old_transcript,
                );
                let mut new_proof = verify_proof_with_instance_commitments(
                    &params,
                    &vk,
                    BatchStrategy::new(&params, rho),
                    commitments.next().unwrap(),
                    &mut new_transcript,
                );
                if let Ok(msm) = &mut new_proof {
                    msm.extract_known(&known);
                }
                assert_eq!(
                    format!("{:?}", old_proof.as_ref().map(|_| ())),
                    format!("{:?}", new_proof.as_ref().map(|_| ()))
                );
                assert_eq!(
                    old_transcript.squeeze_challenge().get_scalar(),
                    new_transcript.squeeze_challenge().get_scalar(),
                );
                match (old_proof, new_proof) {
                    (Ok(old_proof), Ok(new_proof)) => {
                        old.add_msm_batch(old_proof);
                        optimized.add_msm_batch(new_proof);
                    }
                    (Err(_), Err(_)) => break,
                    _ => unreachable!("errors above are identical"),
                }
            }
            assert_eq!(old.fingerprint_terms(), optimized.fingerprint_terms());
            assert_eq!(old.clone().eval(), optimized.eval());
            if altered == 0 {
                assert!(old.eval());
            } else if altered == 1 {
                assert!(!old.eval());
            }
        }
    }
}
