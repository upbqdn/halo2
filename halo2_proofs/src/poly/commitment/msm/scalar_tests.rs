use super::CanonicalFieldKey;
use crate::poly::commitment::Params;
use ff::{Field, PrimeField};
use group::Group;
use pasta_curves::{
    arithmetic::{CurveAffine, CurveExt},
    EpAffine, EqAffine, Fp, Fq,
};
use rand::rngs::SysRng;
use rand_core::UnwrapErr;

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
fn buffered_owned_msms_match_naive_sums() {
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
                for point in [
                    C::identity(),
                    params.g[1],
                    params.g[1],
                    -params.g[1],
                    params.g[2],
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
                incremental.add_msm(&msm);
                batched.add_msm_batch(msm);
            }
            // Exercise both sources and destinations containing buffered terms.
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
fn known_key_terms_match_the_previous_final_msm() {
    use std::sync::Arc;

    use super::KnownTerms;

    fn check<C: CurveAffine>() {
        let params = Params::<C>::new(4);
        let mut rng = UnwrapErr(SysRng);
        let recurring = [params.g[0], params.g[1], -params.g[0], params.g[1]];
        let known = Arc::new(KnownTerms::new(recurring));
        // Includes single proofs, parallel reduction shapes, repeated and negated
        // commitments, unknown collisions, and both accepting/rejecting MSMs.
        for count in [1, 2, 3, 4, 8, 16] {
            for tampered in [false, true] {
                let mut old = params.empty_msm();
                let mut optimized = params.empty_msm();
                for _ in 0..count {
                    let mut proof = params.empty_msm();
                    proof.add_to_g_scalars(
                        &(0..params.n)
                            .map(|_| C::Scalar::random(&mut rng))
                            .collect::<Vec<_>>(),
                    );
                    proof.add_to_w_scalar(C::Scalar::random(&mut rng));
                    proof.add_to_u_scalar(C::Scalar::random(&mut rng));
                    for point in recurring.into_iter().chain([
                        params.g[2],
                        -params.g[2],
                        params.w,
                        C::identity(),
                    ]) {
                        proof.append_term(C::Scalar::random(&mut rng), point);
                    }
                    old.add_msm_batch(proof.clone());
                    proof.extract_known(&known);
                    optimized.add_msm_batch(proof);
                }
                // Make one MSM exactly zero and the other nonzero by an unrelated
                // base; this checks both final evaluation results, not just terms.
                let old_sum = {
                    let (g, w, u, other) = old.fingerprint_terms();
                    let mut sum = C::Curve::identity();
                    for (scalar, base) in g.unwrap().iter().zip(&params.g) {
                        sum += *base * scalar;
                    }
                    sum += params.w * w.unwrap();
                    sum += params.u * u.unwrap();
                    for (scalar, x, y) in other {
                        sum += C::from_xy(x, y).unwrap() * scalar;
                    }
                    sum
                };
                let correction = (-old_sum).to_affine_vartime();
                old.append_term(C::Scalar::ONE, correction);
                optimized.append_term(C::Scalar::ONE, correction);
                if tampered {
                    old.append_term(C::Scalar::ONE, params.g[3]);
                    optimized.append_term(C::Scalar::ONE, params.g[3]);
                }
                assert_eq!(old.fingerprint_terms(), optimized.fingerprint_terms());
                assert_eq!(old.clone().eval(), !tampered);
                assert_eq!(optimized.eval(), !tampered);
            }
        }
    }
    check::<EpAffine>();
    check::<EqAffine>();
}
