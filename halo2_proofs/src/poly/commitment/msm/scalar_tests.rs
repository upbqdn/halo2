use super::{CanonicalFieldKey, KnownTerms};
use crate::poly::commitment::Params;
use ff::{Field, PrimeField};
use group::Group;
use pasta_curves::{
    arithmetic::{CurveAffine, CurveExt},
    EpAffine, EqAffine, Fp, Fq,
};
use rand::rngs::SysRng;
use rand_core::UnwrapErr;
use std::sync::Arc;

#[test]
fn cached_keys_preserve_pasta_order() {
    fn check<F: PrimeField + Ord>() {
        let mut rng = UnwrapErr(SysRng);
        let mut values = vec![F::ZERO, F::ONE, -F::ONE];
        values.extend((0..32).map(|_| F::random(&mut rng)));
        for a in &values {
            for b in &values {
                assert_eq!(
                    CanonicalFieldKey::new(*a).cmp(&CanonicalFieldKey::new(*b)),
                    a.cmp(b)
                );
            }
        }
    }
    check::<Fp>();
    check::<Fq>();
}

#[test]
fn append_only_msms_match_naive_sums_and_bound_storage() {
    fn check<C: CurveAffine>() {
        let params = Params::<C>::new(3);
        let mut rng = UnwrapErr(SysRng);
        for count in [0, 1, 2, 17] {
            let mut incremental = params.empty_msm();
            let mut batched = params.empty_msm();
            let mut expected = C::Curve::identity();
            for _ in 0..count {
                let mut msm = params.empty_msm();
                let scalars = (0..params.n)
                    .map(|_| C::Scalar::random(&mut rng))
                    .collect::<Vec<_>>();
                for (base, scalar) in params.g.iter().zip(&scalars) {
                    expected += *base * scalar;
                }
                let allocation = scalars.as_ptr();
                msm.add_owned_g_scalars(scalars);
                assert_eq!(msm.g_scalars.as_ref().unwrap().as_ptr(), allocation);
                msm.add_to_w_scalar(C::Scalar::ONE);
                msm.add_to_u_scalar(-C::Scalar::ONE);
                expected += params.w - params.u;
                let random_point = loop {
                    let point = (params.g[0] * C::Scalar::random(&mut rng)).to_affine_vartime();
                    if !bool::from(point.is_identity()) {
                        break point;
                    }
                };
                for point in [
                    C::identity(),
                    params.g[1],
                    params.g[1],
                    -params.g[1],
                    params.g[2],
                    random_point,
                    -random_point,
                    C::identity(),
                ] {
                    for scalar in [
                        C::Scalar::ZERO,
                        C::Scalar::ONE,
                        -C::Scalar::ONE,
                        C::Scalar::random(&mut rng),
                    ] {
                        msm.append_term(scalar, point);
                        expected += point * scalar;
                    }
                }
                assert_eq!(msm.other.len(), 24);
                incremental.add_msm(&msm);
                batched.add_msm_batch(msm);
            }
            assert_eq!(incremental.other.len(), count * 24);
            assert_eq!(batched.other.len(), count * 24);
            assert_eq!(incremental.clone().eval_point(), expected);
            assert_eq!(batched.clone().eval_point(), expected);
            // Exercise both borrowed and owned additions with existing arbitrary terms.
            let mut nested = params.empty_msm();
            nested.add_msm_batch(batched.clone());
            let mut borrowed = params.empty_msm();
            borrowed.add_msm(&nested);
            for scale in [
                C::Scalar::ZERO,
                C::Scalar::ONE,
                -C::Scalar::ONE,
                C::Scalar::random(&mut rng),
            ] {
                for mut msm in [
                    incremental.clone(),
                    batched.clone(),
                    nested.clone(),
                    borrowed.clone(),
                ] {
                    msm.scale(scale);
                    msm.append_term(-C::Scalar::ONE, (expected * scale).to_affine_vartime());
                    assert!(msm.clone().eval());
                    msm.append_term(C::Scalar::ONE, params.g[0]);
                    assert!(!msm.eval());
                }
            }
        }
    }
    check::<EpAffine>();
    check::<EqAffine>();
}

#[test]
fn known_terms_match_canonical_batch_for_duplicate_and_negated_bases() {
    fn check<C: CurveAffine>() {
        let params = Params::<C>::new(3);
        let p = params.g[1];
        let q = params.g[2];
        let unknown = params.g[3];
        for known in [
            Arc::new(KnownTerms::new([p, -p, p, q, C::identity()])),
            Arc::new(KnownTerms::new([-p, p, q])),
        ] {
            assert_eq!(known.points.len(), 2);

            for count in [2, 2049] {
                let mut plain = params.empty_msm();
                let mut indexed = params.empty_msm();
                for i in 0..count {
                    let mut proof = params.empty_msm();
                    if i % 2 == 0 {
                        proof.append_term(C::Scalar::from(2), p);
                        proof.append_term(C::Scalar::from(3), -p);
                    } else {
                        proof.append_term(C::Scalar::from(3), -p);
                        proof.append_term(C::Scalar::from(2), p);
                    }
                    proof.append_term(C::Scalar::from(5), q);
                    proof.append_term(C::Scalar::from(7), unknown);
                    proof.append_term(C::Scalar::from(11), -unknown);
                    proof.scale(C::Scalar::from(3));
                    let mut extracted = proof.clone();
                    extracted.extract_known(&known);
                    assert_eq!(extracted.other.len(), 2);
                    plain.add_msm_batch(proof);
                    indexed.add_msm_batch(extracted);
                }
                assert_eq!(plain.other.len(), count * 5);
                assert_eq!(indexed.other.len(), count * 2);
                assert_eq!(indexed.known.as_ref().unwrap().1.len(), known.points.len());
                assert_eq!(plain.clone().eval_point(), indexed.clone().eval_point());
                assert_eq!(plain.eval(), indexed.eval());
            }
        }
    }
    check::<EpAffine>();
    check::<EqAffine>();
}
