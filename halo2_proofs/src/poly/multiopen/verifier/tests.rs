use super::*;
use crate::arithmetic::{eval_polynomial, lagrange_interpolate};
use pasta_curves::{Fp, Fq};
use rand::rngs::SysRng;
use rand_core::UnwrapErr;

fn reference<F: VartimeField>(
    point_sets: &[Vec<F>],
    eval_sets: &[Vec<F>],
    proof_evals: &[F],
    x_2: F,
    x_3: F,
) -> F {
    point_sets.iter().zip(eval_sets).zip(proof_evals).fold(
        F::ZERO,
        |acc, ((points, evals), proof_eval)| {
            let r_eval = eval_polynomial(&lagrange_interpolate(points, evals), x_3);
            let eval = points.iter().fold(*proof_eval - r_eval, |eval, point| {
                eval * (x_3 - point).invert_vartime().unwrap()
            });
            acc * x_2 + eval
        },
    )
}

#[test]
fn batched_denominators_match_polynomial_interpolation() {
    fn check<F: VartimeField + From<u64>>() {
        let mut rng = UnwrapErr(SysRng);
        assert_eq!(
            compute_msm_eval::<F>(&[], &[], &[], F::ZERO, F::ONE),
            F::ZERO
        );
        let points = [F::ZERO, F::ONE, -F::ONE, F::from(2)];
        for count in 0..=4 {
            let sets = (0..=count)
                .map(|n| points[..n].to_vec())
                .collect::<Vec<_>>();
            for _ in 0..16 {
                let evals = sets
                    .iter()
                    .map(|points| points.iter().map(|_| F::random(&mut rng)).collect())
                    .collect::<Vec<Vec<_>>>();
                let proof_evals = sets.iter().map(|_| F::random(&mut rng)).collect::<Vec<_>>();
                for x_2 in [F::ZERO, F::ONE, -F::ONE, F::random(&mut rng)] {
                    let mut x_3 = F::random(&mut rng);
                    while points.contains(&x_3) {
                        x_3 = F::random(&mut rng);
                    }
                    assert_eq!(
                        compute_msm_eval(&sets, &evals, &proof_evals, x_2, x_3),
                        reference(&sets, &evals, &proof_evals, x_2, x_3),
                    );
                }
            }
        }
    }
    check::<Fp>();
    check::<Fq>();
}

#[test]
#[should_panic(expected = "verifier challenge must differ")]
fn challenge_at_query_point_preserves_zero_denominator_failure() {
    compute_msm_eval(
        &[vec![Fp::ONE]],
        &[vec![Fp::ONE]],
        &[Fp::ONE],
        Fp::ONE,
        Fp::ONE,
    );
}

// The pre-optimization verifier path, retained as a differential oracle.
fn reference_verify_proof<'r, 'params: 'r, I, C, E, T>(
    params: &'params Params<C>,
    transcript: &mut T,
    queries: I,
    mut msm: MSM<'params, C>,
) -> Result<Guard<'params, C, E>, Error>
where
    I: IntoIterator<Item = VerifierQuery<'r, 'params, C>> + Clone,
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptRead<C, E>,
{
    let x_1: ChallengeX1<_> = transcript.squeeze_challenge_scalar();
    let x_2: ChallengeX2<_> = transcript.squeeze_challenge_scalar();
    let (commitment_map, point_sets) =
        construct_intermediate_sets(queries).ok_or(Error::OpeningError)?;
    let mut q_commitments: Vec<_> = vec![(params.empty_msm(), C::Scalar::ONE); point_sets.len()];
    let mut q_eval_sets = point_sets
        .iter()
        .map(|points| vec![C::Scalar::ZERO; points.len()])
        .collect::<Vec<_>>();
    for data in commitment_map.into_iter().rev() {
        let (q_commitment, x_1_power) = &mut q_commitments[data.set_index];
        match data.commitment {
            CommitmentReference::Commitment(c) => q_commitment.append_term(*x_1_power, *c),
            CommitmentReference::MSM(inner) => {
                let mut inner = inner.clone();
                inner.scale(*x_1_power);
                q_commitment.add_msm(&inner);
            }
        }
        for (eval, set_eval) in data.evals.iter().zip(&mut q_eval_sets[data.set_index]) {
            *set_eval += *eval * *x_1_power;
        }
        *x_1_power *= *x_1;
    }
    let q_prime_commitment = transcript.read_point().map_err(|_| Error::SamplingError)?;
    let x_3: ChallengeX3<_> = transcript.squeeze_challenge_scalar();
    let mut u = Vec::with_capacity(q_eval_sets.len());
    for _ in 0..q_eval_sets.len() {
        u.push(transcript.read_scalar().map_err(|_| Error::SamplingError)?);
    }
    let msm_eval = compute_msm_eval(&point_sets, &q_eval_sets, &u, *x_2, *x_3);
    let x_4: ChallengeX4<_> = transcript.squeeze_challenge_scalar();
    msm.append_term(C::Scalar::ONE, q_prime_commitment);
    let (msm, v) = q_commitments.into_iter().zip(u.iter()).fold(
        (msm, msm_eval),
        |(mut msm, msm_eval), ((q_commitment, _), q_eval)| {
            msm.scale(*x_4);
            msm.add_msm(&q_commitment);
            (msm, msm_eval * &(*x_4) + q_eval)
        },
    );
    super::super::commitment::verify_proof(params, msm, transcript, *x_3, v)
}

fn compare_paths(
    params: &Params<crate::pasta::EqAffine>,
    proof: &[u8],
    queries: Vec<VerifierQuery<'_, '_, crate::pasta::EqAffine>>,
    expected: Option<bool>,
) {
    compare_paths_with_seed(params, proof, queries, expected, params.empty_msm());
}

fn compare_paths_with_seed<'params>(
    params: &'params Params<crate::pasta::EqAffine>,
    proof: &[u8],
    queries: Vec<VerifierQuery<'_, 'params, crate::pasta::EqAffine>>,
    expected: Option<bool>,
    seed: MSM<'params, crate::pasta::EqAffine>,
) {
    use crate::transcript::{Blake2bRead, Challenge255, Transcript};
    use std::io::Cursor;

    let run = |reference: bool| {
        let mut reader = Cursor::new(proof);
        let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&mut reader);
        let result = if reference {
            reference_verify_proof(params, &mut transcript, queries.clone(), seed.clone())
        } else {
            verify_proof(params, &mut transcript, queries.clone(), seed.clone())
        };
        let challenge = *transcript.squeeze_challenge_scalar::<()>();
        (result, reader.position(), challenge)
    };
    let (fast, fast_bytes, fast_challenge) = run(false);
    let (original, original_bytes, original_challenge) = run(true);
    assert_eq!(fast_bytes, original_bytes);
    assert_eq!(fast_challenge, original_challenge);
    match (fast, original) {
        (Ok(fast), Ok(original)) => {
            let fast = fast.use_challenges();
            let original = original.use_challenges();
            let mut difference = fast.clone();
            difference.scale(-Fp::ONE);
            difference.add_msm(&original);
            assert!(difference.eval(), "verifier MSMs differ as group elements");
            let fast_valid = fast.eval();
            let original_valid = original.eval();
            assert_eq!(fast_valid, original_valid);
            assert_eq!(Some(fast_valid), expected);
        }
        (Err(fast), Err(original)) => {
            assert_eq!(format!("{fast:?}"), format!("{original:?}"));
            assert_eq!(expected, None);
        }
        _ => panic!("verifier paths returned different outcomes"),
    }
}

#[test]
fn verifier_sets_match_generic_ordering_and_duplicates() {
    use crate::pasta::EqAffine;
    use group::{Curve, Group};

    let mut rng = UnwrapErr(SysRng);
    let commitments: Vec<_> = (0..9)
        .map(|_| crate::pasta::Eq::random(&mut rng).to_affine())
        .collect();
    let points = (0..8).map(Fp::from).collect::<Vec<_>>();
    for length in 0..=8 {
        let queries = (0..length)
            .rev()
            .map(|i| {
                VerifierQuery::<EqAffine>::new_commitment(
                    &commitments[i % 3],
                    points[i],
                    Fp::from(i as u64 + 11),
                )
            })
            .collect::<Vec<_>>();
        let (generic, generic_sets) = construct_intermediate_sets(queries.clone()).unwrap();
        let (fast, fast_sets) = construct_verifier_sets(queries.clone()).unwrap().unwrap();
        assert_eq!(generic_sets, fast_sets);
        for (data, (commitment, set, evals)) in generic.iter().zip(fast.iter()) {
            assert!(data.commitment == *commitment);
            assert_eq!(data.set_index, *set);
            assert_eq!(data.evals, evals.evals[..evals.len]);
        }
        assert_eq!(generic.len(), fast.len());
        if let Some(first) = queries.first() {
            let mut duplicated = queries.clone();
            duplicated.push(first.clone());
            assert!(construct_intermediate_sets(duplicated.clone()).is_none());
            assert!(matches!(construct_verifier_sets(duplicated), Ok(None)));
        }
    }
    let equal_value_different_address = commitments[0];
    let queries = vec![
        VerifierQuery::new_commitment(&commitments[0], points[0], Fp::from(1)),
        VerifierQuery::new_commitment(&commitments[1], points[1], Fp::from(2)),
        VerifierQuery::new_commitment(&commitments[0], points[1], Fp::from(3)),
        VerifierQuery::new_commitment(&commitments[1], points[0], Fp::from(4)),
        VerifierQuery::new_commitment(&equal_value_different_address, points[0], Fp::from(5)),
    ];
    let (generic, generic_sets) = construct_intermediate_sets(queries.clone()).unwrap();
    let (fast, fast_sets) = construct_verifier_sets(queries).unwrap().unwrap();
    assert_eq!(generic_sets, fast_sets);
    assert_eq!(generic.len(), fast.len());
    for (data, (commitment, set, evals)) in generic.iter().zip(fast.iter()) {
        assert!(data.commitment == *commitment);
        assert_eq!(data.set_index, *set);
        assert_eq!(data.evals, evals.evals[..evals.len]);
    }
}

#[test]
fn randomized_verifier_sets_match_generic_construction() {
    use crate::pasta::EqAffine;
    use group::{Curve, Group};
    use rand::RngExt;

    let mut rng = UnwrapErr(SysRng);
    let commitments: Vec<EqAffine> = (0..12)
        .map(|_| crate::pasta::Eq::random(&mut rng).to_affine())
        .collect();
    let points: Vec<Fp> = (0..5).map(|_| Fp::random(&mut rng)).collect();
    for _ in 0..200 {
        let mut queries = Vec::new();
        for _ in 0..rng.random_range(1..40) {
            let c = &commitments[rng.random_range(0..commitments.len())];
            let p = points[rng.random_range(0..points.len())];
            queries.push(VerifierQuery::new_commitment(c, p, Fp::random(&mut rng)));
        }
        let generic = construct_intermediate_sets(queries.clone());
        let fast = construct_verifier_sets(queries).expect("five points fit inline");
        match (generic, fast) {
            (None, None) => {}
            (Some((map, sets)), Some((fast_map, fast_sets))) => {
                assert_eq!(sets, fast_sets);
                assert_eq!(map.len(), fast_map.len());
                for (data, (commitment, set, evals)) in map.iter().zip(&fast_map) {
                    assert!(data.commitment == *commitment);
                    assert_eq!(data.set_index, *set);
                    assert_eq!(&data.evals[..], &evals.evals[..evals.len]);
                }
            }
            (generic, fast) => panic!("{} vs {}", generic.is_some(), fast.is_some()),
        }
    }
}

#[test]
fn real_multiopen_proofs_match_reference_verifier() {
    use crate::arithmetic::eval_polynomial;
    use crate::pasta::{EqAffine, Fp};
    use crate::poly::{commitment::Blind, multiopen::ProverQuery, EvaluationDomain};
    use crate::transcript::{Blake2bRead, Blake2bWrite, Challenge255, Transcript};
    use group::{ff::PrimeField, Curve};

    const K: u32 = 4;
    let params: Params<EqAffine> = Params::new(K);
    let domain = EvaluationDomain::new(1, K);
    let mut rng = UnwrapErr(SysRng);
    let mut poly = domain.empty_coeff();
    let mut other = domain.empty_coeff();
    for (i, (a, b)) in poly.iter_mut().zip(other.iter_mut()).enumerate() {
        *a = Fp::from(3 + i as u64);
        *b = Fp::from(12 + i as u64);
    }
    let third_poly = poly.clone();
    let blind = Blind(Fp::random(&mut rng));
    let a = params.commit(&poly, blind).to_affine();
    let b = params.commit(&other, blind).to_affine();
    let c = params.commit(&third_poly, blind).to_affine();
    let mut b_msm = params.empty_msm();
    b_msm.append_term(Fp::ONE, b);
    let x = Fp::from(17);
    let y = Fp::from(23);
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    super::super::create_proof(
        &params,
        &mut rng,
        &mut transcript,
        [
            ProverQuery {
                point: x,
                poly: &poly,
                blind,
            },
            ProverQuery {
                point: x,
                poly: &other,
                blind,
            },
            ProverQuery {
                point: x,
                poly: &third_poly,
                blind,
            },
            ProverQuery {
                point: y,
                poly: &other,
                blind,
            },
            ProverQuery {
                point: y,
                poly: &third_poly,
                blind,
            },
        ],
    )
    .unwrap();
    let proof = transcript.finalize();
    let queries = || {
        vec![
            VerifierQuery::new_commitment(&a, x, eval_polynomial(&poly, x)),
            VerifierQuery::new_msm(&b_msm, x, eval_polynomial(&other, x)),
            VerifierQuery::new_commitment(&c, x, eval_polynomial(&third_poly, x)),
            VerifierQuery::new_msm(&b_msm, y, eval_polynomial(&other, y)),
            VerifierQuery::new_commitment(&c, y, eval_polynomial(&third_poly, y)),
        ]
    };
    compare_paths(&params, &proof, queries(), Some(true));
    let mut seed = params.empty_msm();
    seed.add_to_w_scalar(Fp::from(7));
    compare_paths_with_seed(&params, &proof, queries(), Some(false), seed);
    let mut bad = queries();
    bad[2].eval += Fp::ONE;
    compare_paths(&params, &proof, bad, Some(false));
    let mut tampered_proof = proof.clone();
    // The first quotient evaluation follows the quotient commitment (one point).
    tampered_proof[32..64].copy_from_slice(Fp::from(999).to_repr().as_ref());
    assert_ne!(tampered_proof, proof);
    compare_paths(&params, &tampered_proof, queries(), Some(false));

    // Force the Lagrange vanishing denominator to zero in both paths.
    let mut read = Blake2bRead::<_, EqAffine, Challenge255<EqAffine>>::init(&proof[..]);
    let _: ChallengeX1<_> = read.squeeze_challenge_scalar();
    let _: ChallengeX2<_> = read.squeeze_challenge_scalar();
    read.read_point().unwrap();
    let x_3: ChallengeX3<_> = read.squeeze_challenge_scalar();
    for reference in [false, true] {
        let run = || {
            let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
            let query = vec![VerifierQuery::new_commitment(&a, *x_3, Fp::ONE)];
            if reference {
                let _ = reference_verify_proof(&params, &mut transcript, query, params.empty_msm());
            } else {
                let _ = verify_proof(&params, &mut transcript, query, params.empty_msm());
            }
        };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err());
    }

    let mut duplicated = queries();
    duplicated.push(duplicated[0].clone());
    compare_paths(&params, &proof, duplicated, None);
    compare_paths(&params, &[], vec![], None);

    // The existing multiopen roundtrip uses polynomial queries at multiple points.
    // Nine evaluations of one commitment force the generic fallback.
    let points = (1..=9).map(Fp::from).collect::<Vec<_>>();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    super::super::create_proof(
        &params,
        &mut rng,
        &mut transcript,
        points.iter().map(|point| ProverQuery {
            point: *point,
            poly: &poly,
            blind,
        }),
    )
    .unwrap();
    let proof = transcript.finalize();
    let queries = || {
        points
            .iter()
            .map(|point| VerifierQuery::new_commitment(&a, *point, eval_polynomial(&poly, *point)))
            .collect::<Vec<_>>()
    };
    assert!(construct_verifier_sets(queries()).is_err());
    compare_paths(&params, &proof, queries(), Some(true));
    let mut bad = queries();
    bad[8].eval += Fp::ONE;
    compare_paths(&params, &proof, bad, Some(false));

    // More than one machine word of distinct points also takes the generic path.
    let count = usize::try_from(usize::BITS).unwrap() + 1;
    let polys = vec![poly.clone(); count];
    let commitments = vec![a; count]; // Same group value, distinct commitment addresses.
    let points = (0..count)
        .map(|i| Fp::from(u64::try_from(i).unwrap()))
        .collect::<Vec<_>>();
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    super::super::create_proof(
        &params,
        &mut rng,
        &mut transcript,
        points.iter().enumerate().map(|(i, point)| ProverQuery {
            point: *point,
            poly: &polys[i],
            blind,
        }),
    )
    .unwrap();
    let proof = transcript.finalize();
    let large = || {
        points
            .iter()
            .zip(&commitments)
            .map(|(point, c)| {
                VerifierQuery::new_commitment(c, *point, eval_polynomial(&poly, *point))
            })
            .collect::<Vec<_>>()
    };
    assert!(construct_verifier_sets(large()).is_err());
    compare_paths(&params, &proof, large(), Some(true));
    let mut bad = large();
    bad[count - 1].eval += Fp::ONE;
    compare_paths(&params, &proof, bad, Some(false));
}
