use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use ff::Field;
use pasta_curves::arithmetic::{VartimeBatchInvert, VartimeField};

use super::super::{
    commitment::{Guard, Params, MSM},
    Error,
};
use super::{
    construct_intermediate_sets, ChallengeX1, ChallengeX2, ChallengeX3, ChallengeX4,
    CommitmentReference, Query, VerifierQuery,
};
use crate::arithmetic::CurveAffine;
use crate::transcript::{EncodedChallenge, TranscriptRead};

/// Evaluates the multi-point quotient with one inversion for all Lagrange
/// denominators and vanishing polynomials, without allocating interpolants.
fn compute_msm_eval<F: VartimeField>(
    point_sets: &[Vec<F>],
    q_eval_sets: &[Vec<F>],
    proof_evals: &[F],
    x_2: F,
    x_3: F,
) -> F {
    assert_eq!(point_sets.len(), q_eval_sets.len());
    assert_eq!(point_sets.len(), proof_evals.len());
    let count = point_sets.iter().map(|points| points.len() + 1).sum();
    let mut denominators = Vec::with_capacity(count);
    for points in point_sets {
        for (j, x_j) in points.iter().enumerate() {
            let denominator = points
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != j)
                .fold(F::ONE, |acc, (_, x_k)| acc * (*x_j - x_k));
            assert!(
                !denominator.is_zero_vartime(),
                "point sets must contain distinct points"
            );
            denominators.push(denominator);
        }
        let vanishing = points.iter().fold(F::ONE, |acc, point| acc * (x_3 - point));
        // Preserve the previous verifier's failure on a zero denominator.
        assert!(
            !vanishing.is_zero_vartime(),
            "verifier challenge must differ from query points"
        );
        denominators.push(vanishing);
    }
    denominators.iter_mut().batch_invert_vartime();
    let mut inverses = denominators.into_iter();
    point_sets.iter().zip(q_eval_sets).zip(proof_evals).fold(
        F::ZERO,
        |acc, ((points, evals), proof_eval)| {
            assert_eq!(points.len(), evals.len());
            let r_eval =
                points
                    .iter()
                    .enumerate()
                    .zip(evals)
                    .fold(F::ZERO, |r_eval, ((j, _), eval)| {
                        let numerator = points
                            .iter()
                            .enumerate()
                            .filter(|(k, _)| *k != j)
                            .fold(F::ONE, |acc, (_, point)| acc * (x_3 - point));
                        let inverse = inverses
                            .next()
                            .expect("one denominator per interpolation point");
                        r_eval + *eval * numerator * inverse
                    });
            let inverse = inverses
                .next()
                .expect("one vanishing denominator per point set");
            acc * x_2 + (*proof_eval - r_eval) * inverse
        },
    )
}

// Adapted from Zakura's verifier-specific query sets (MIT OR Apache-2.0).
const INLINE_EVALS: usize = 8;

// CommitmentReference hashes pointer addresses; no bytewise hashing is needed.
#[derive(Default)]
struct AddressHasher(u64);

impl Hasher for AddressHasher {
    fn finish(&self) -> u64 {
        self.0.rotate_left(32)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0.rotate_left(8) ^ u64::from(byte)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }

    fn write_usize(&mut self, value: usize) {
        self.0 = (self.0 ^ u64::try_from(value).expect("pointer fits in u64"))
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
}

enum Evals<F> {
    Inline(InlineEvals<F>),
    Heap(Vec<F>),
}

impl<F> Evals<F> {
    fn as_slice(&self) -> &[F] {
        match self {
            Evals::Inline(evals) => &evals.evals[..evals.len],
            Evals::Heap(evals) => evals,
        }
    }
}

struct InlineEvals<F> {
    len: usize,
    points: [u8; INLINE_EVALS],
    evals: [F; INLINE_EVALS],
}

// Preserve the generic constructor's first-seen commitment, point, and set ordering.
// An unrepresentable shape falls back to construct_intermediate_sets.
#[allow(clippy::type_complexity)]
fn construct_verifier_sets<'r, 'params: 'r, C: CurveAffine, I>(
    queries: I,
) -> Result<
    Option<(
        Vec<(
            CommitmentReference<'r, 'params, C>,
            usize,
            InlineEvals<C::Scalar>,
        )>,
        Vec<Vec<C::Scalar>>,
    )>,
    (),
>
where
    I: IntoIterator<Item = VerifierQuery<'r, 'params, C>>,
{
    let queries = queries.into_iter();
    let capacity = queries.size_hint().0.min(256);
    let mut points = Vec::new();
    let mut commitments = Vec::with_capacity(capacity);
    #[allow(clippy::mutable_key_type)]
    let mut indices: HashMap<_, _, BuildHasherDefault<AddressHasher>> =
        HashMap::with_capacity_and_hasher(capacity, Default::default());

    for query in queries {
        let point = query.get_point();
        let point_index = match points.iter().position(|p| *p == point) {
            Some(index) => index,
            None => {
                if points.len() == usize::BITS as usize {
                    return Err(());
                }
                points.push(point);
                points.len() - 1
            }
        };
        let commitment = query.get_commitment();
        let index = *indices.entry(commitment).or_insert_with(|| {
            commitments.push((
                commitment,
                0usize,
                InlineEvals {
                    len: 0,
                    points: [0; INLINE_EVALS],
                    evals: [C::Scalar::ZERO; INLINE_EVALS],
                },
            ));
            commitments.len() - 1
        });
        let (_, mask, evals) = &mut commitments[index];
        let bit = 1usize << point_index;
        if *mask & bit != 0 {
            return Ok(None);
        }
        if evals.len == INLINE_EVALS {
            return Err(());
        }
        *mask |= bit;
        let mut at = evals.len;
        while at > 0 && usize::from(evals.points[at - 1]) > point_index {
            evals.points[at] = evals.points[at - 1];
            evals.evals[at] = evals.evals[at - 1];
            at -= 1;
        }
        evals.points[at] = u8::try_from(point_index).map_err(|_| ())?;
        evals.evals[at] = query.get_eval();
        evals.len += 1;
    }

    let mut masks = Vec::new();
    for (_, mask, _) in &mut commitments {
        *mask = match masks.iter().position(|m| m == mask) {
            Some(set) => set,
            None => {
                masks.push(*mask);
                masks.len() - 1
            }
        };
    }
    let point_sets = masks
        .into_iter()
        .map(|mask| {
            points
                .iter()
                .enumerate()
                .filter_map(|(i, p)| ((mask >> i) & 1 == 1).then_some(*p))
                .collect()
        })
        .collect();
    Ok(Some((commitments, point_sets)))
}

/// Verify a multi-opening proof
pub fn verify_proof<
    'r,
    'params: 'r,
    I,
    C: CurveAffine,
    E: EncodedChallenge<C>,
    T: TranscriptRead<C, E>,
>(
    params: &'params Params<C>,
    transcript: &mut T,
    queries: I,
    mut msm: MSM<'params, C>,
) -> Result<Guard<'params, C, E>, Error>
where
    I: IntoIterator<Item = VerifierQuery<'r, 'params, C>> + Clone,
{
    // Sample x_1 for compressing openings at the same point sets together
    let x_1: ChallengeX1<_> = transcript.squeeze_challenge_scalar();

    // Sample a challenge x_2 for keeping the multi-point quotient
    // polynomial terms linearly independent.
    let x_2: ChallengeX2<_> = transcript.squeeze_challenge_scalar();

    #[allow(clippy::type_complexity)]
    let (commitment_map, point_sets): (Vec<(_, usize, Evals<C::Scalar>)>, _) =
        match construct_verifier_sets(queries.clone()) {
            Ok(sets) => {
                let (map, sets) = sets.ok_or(Error::OpeningError)?;
                (
                    map.into_iter()
                        .map(|(c, set, evals)| (c, set, Evals::Inline(evals)))
                        .collect(),
                    sets,
                )
            }
            Err(()) => {
                let (map, sets) =
                    construct_intermediate_sets(queries).ok_or(Error::OpeningError)?;
                (
                    map.into_iter()
                        .map(|data| (data.commitment, data.set_index, Evals::Heap(data.evals)))
                        .collect(),
                    sets,
                )
            }
        };

    // Defer combining commitment terms until x_4 has been sampled.
    let mut q_commitments: Vec<_> = vec![
        (Vec::new(), C::Scalar::ONE); // (commitments with x_1 powers, next x_1 power).
        point_sets.len()];

    let mut q_eval_sets = Vec::with_capacity(point_sets.len());
    for point_set in &point_sets {
        q_eval_sets.push(vec![C::Scalar::ZERO; point_set.len()]);
    }

    {
        let mut accumulate = |set_idx: usize, new_commitment, evals: &[C::Scalar]| {
            let (q_commitment, x_1_power) = &mut q_commitments[set_idx];
            q_commitment.push((new_commitment, *x_1_power));
            for (eval, set_eval) in evals.iter().zip(q_eval_sets[set_idx].iter_mut()) {
                *set_eval += (*eval) * (*x_1_power);
            }
            *x_1_power *= *x_1;
        };

        // Reverse first-seen commitment order gives increasing x_1 powers per set.
        for (commitment, set_index, evals) in commitment_map.iter().rev() {
            accumulate(*set_index, *commitment, evals.as_slice());
        }
    }

    // Obtain the commitment to the multi-point quotient polynomial f(X).
    let q_prime_commitment = transcript.read_point().map_err(|_| Error::SamplingError)?;

    // Sample a challenge x_3 for checking that f(X) was committed to
    // correctly.
    let x_3: ChallengeX3<_> = transcript.squeeze_challenge_scalar();

    // u is a vector containing the evaluations of the Q polynomial
    // commitments at x_3
    let mut u = Vec::with_capacity(q_eval_sets.len());
    for _ in 0..q_eval_sets.len() {
        u.push(transcript.read_scalar().map_err(|_| Error::SamplingError)?);
    }

    // We can compute the expected msm_eval at x_3 using the u provided
    // by the prover and from x_2
    let msm_eval = compute_msm_eval(&point_sets, &q_eval_sets, &u, *x_2, *x_3);

    // Sample a challenge x_4 that we will use to collapse the openings of
    // the various remaining polynomials at x_3 together.
    let x_4: ChallengeX4<_> = transcript.squeeze_challenge_scalar();

    // Each x_1 power and x_4 power is applied once to a commitment term.
    let mut x_4_power = C::Scalar::ONE;
    let mut v = msm_eval;
    let mut x_4_powers = Vec::with_capacity(q_commitments.len());
    for _ in &u {
        x_4_powers.push(x_4_power);
        x_4_power *= *x_4;
    }
    for q_eval in &u {
        v = v * &(*x_4) + q_eval;
    }
    msm.scale(x_4_power);
    msm.append_term(x_4_power, q_prime_commitment);
    for ((q_commitment, _), x_4_power) in
        q_commitments.into_iter().zip(x_4_powers.into_iter().rev())
    {
        for (commitment, x_1_power) in q_commitment {
            let scalar = x_1_power * x_4_power;
            match commitment {
                CommitmentReference::Commitment(c) => msm.append_term(scalar, *c),
                CommitmentReference::MSM(inner) => {
                    let mut inner = inner.clone();
                    inner.scale(scalar);
                    msm.add_msm(&inner);
                }
            }
        }
    }

    // Verify the opening proof
    super::commitment::verify_proof(params, msm, transcript, *x_3, v)
}

impl<'a, 'b, C: CurveAffine> Query<C::Scalar> for VerifierQuery<'a, 'b, C> {
    type Commitment = CommitmentReference<'a, 'b, C>;
    type Eval = C::Scalar;

    fn get_point(&self) -> C::Scalar {
        self.point
    }
    fn get_eval(&self) -> C::Scalar {
        self.eval
    }
    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }
}

#[cfg(test)]
mod tests;
