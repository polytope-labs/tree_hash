//! Merkle proofs over progressive containers.
//!
//! [`ProgressiveMerkleHasher`](crate::ProgressiveMerkleHasher) streams chunks and keeps only the
//! state it needs for the root, so it cannot answer "give me the branch for field `n`". This module
//! rebuilds the same tree shape from a container's field roots and reads sibling paths out of it.
//!
//! # Tree shape
//!
//! A progressive container merkleizes its field roots with `merkleize_progressive` and then mixes
//! in `active_fields`. `merkleize_progressive` is a right leaning spine whose left child at level
//! `k` is an ordinary binary tree over `4^k` leaves:
//!
//! ```text
//!                        root
//!                         /\
//!                        /  \
//!        container_root /    \ active_fields
//!                      /\
//!      level 0 (1)    /  \
//!                        /\
//!      level 1 (4)      /  \
//!                          /\
//!      level 2 (16)       /  \
//!                            /\
//!      level 3 (64)         /  \
//!                              0
//! ```
//!
//! Two consequences the fixed depth scheme does not have. Field depth grows with field index, so
//! branches from one container have different lengths, and the generalized index of a field is not
//! `2^ceil(log2(nfields)) + i`.

use alloc::{vec, vec::Vec};

use crate::{Hash256, BYTES_PER_CHUNK};
use ethereum_hashing::hash32_concat;

/// Errors returned when building a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The requested field index is not present in the supplied field roots.
    FieldIndexOutOfBounds { index: usize, len: usize },
    /// No field roots were supplied, so there is nothing to prove.
    NoFields,
}

/// The number of leaves in the binary subtree at `level`.
const fn level_size(level: usize) -> usize {
    1 << (2 * level)
}

/// The index of the first field held by `level`.
///
/// Levels hold `1, 4, 16, 64, ...` fields, so the first index of level `k` is `(4^k - 1) / 3`.
const fn level_start(level: usize) -> usize {
    (level_size(level) - 1) / 3
}

/// The level holding `field_index`, and the offset of the field within that level.
fn locate(field_index: usize) -> (usize, usize) {
    let mut level = 0;
    while level_start(level + 1) <= field_index {
        level += 1;
    }
    (level, field_index - level_start(level))
}

/// The generalized index of the left (binary subtree) child at `level`, within the progressive
/// tree alone, before `active_fields` is mixed in.
///
/// The spine puts these at `2, 6, 14, 30, ...`, which is `2^(k + 2) - 2`.
const fn level_gindex(level: usize) -> u64 {
    (1u64 << (level + 2)) - 2
}

/// The generalized index of `field_index` in a progressive container.
///
/// This is the index against the container's own root, so it already accounts for the
/// `active_fields` mix in. Note that unlike a fixed depth container, two fields of the same
/// container generally sit at different depths.
///
/// ```
/// # use tree_hash::proof::progressive_container_gindex;
/// // Fields 20, 23 and 24 of a 46 field container.
/// assert_eq!(progressive_container_gindex(20), 367);
/// assert_eq!(progressive_container_gindex(23), 2946);
/// assert_eq!(progressive_container_gindex(24), 2947);
/// ```
pub fn progressive_container_gindex(field_index: usize) -> u64 {
    let (level, offset) = locate(field_index);

    // Position within the progressive tree, whose root is the left child of the final root.
    let within = level_gindex(level) * level_size(level) as u64 + offset as u64;

    // Graft that subtree under the left child of the final root. Stripping the leading one bit and
    // re-attaching it below the root's left child is the same as adding it back one place higher.
    within + (1u64 << within.ilog2())
}

/// The depth of `field_index`, which is the number of nodes in its branch.
pub fn progressive_container_depth(field_index: usize) -> usize {
    let (level, _) = locate(field_index);
    // `2 * level` within the binary subtree, one for the spine sibling at this level, `level` more
    // walking back up the spine, and one for `active_fields`.
    3 * level + 2
}

/// Build a binary merkle tree over `leaves`, zero padding up to `size`.
///
/// Returns the tree bottom up, so `tree[0]` is the padded leaves and `tree.last()` is the root.
fn binary_tree(leaves: &[Hash256], size: usize) -> Vec<Vec<Hash256>> {
    let mut layer: Vec<Hash256> = leaves.to_vec();
    layer.resize(size, Hash256::ZERO);

    let mut tree = vec![layer];
    while tree.last().map_or(0, |l| l.len()) > 1 {
        let last = tree.last().expect("tree is never empty");
        let next = last
            .chunks(2)
            .map(|pair| Hash256::from(hash32_concat(pair[0].as_slice(), pair[1].as_slice())))
            .collect::<Vec<_>>();
        tree.push(next);
    }
    tree
}

/// The binary subtree root for each level covered by `field_roots`.
fn level_roots(field_roots: &[Hash256]) -> Vec<Hash256> {
    let mut roots = Vec::new();
    let mut level = 0;
    while level_start(level) < field_roots.len() {
        let start = level_start(level);
        let end = (start + level_size(level)).min(field_roots.len());
        let tree = binary_tree(&field_roots[start..end], level_size(level));
        roots.push(*tree.last().and_then(|l| l.first()).expect("tree has a root"));
        level += 1;
    }
    roots
}

/// The root of the spine from `level` downwards, which is the sibling hanging off the right of
/// each level. Levels past the end of the container are empty, and hash to zero.
fn rest_root(level_roots: &[Hash256], level: usize) -> Hash256 {
    if level >= level_roots.len() {
        return Hash256::ZERO;
    }
    let below = rest_root(level_roots, level + 1);
    Hash256::from(hash32_concat(
        level_roots[level].as_slice(),
        below.as_slice(),
    ))
}

/// The `merkleize_progressive` root of `field_roots`, before `active_fields` is mixed in.
pub fn progressive_root(field_roots: &[Hash256]) -> Hash256 {
    rest_root(&level_roots(field_roots), 0)
}

/// The full container root, with `active_fields` mixed in.
pub fn progressive_container_root(
    field_roots: &[Hash256],
    active_fields: [u8; BYTES_PER_CHUNK],
) -> Hash256 {
    crate::mix_in_active_fields(&progressive_root(field_roots), active_fields)
}

/// Build the merkle branch proving `field_roots[field_index]` against the container root.
///
/// The branch is ordered leaf first, so it pairs with [`is_valid_merkle_branch`] and with the
/// generalized index from [`progressive_container_gindex`].
pub fn progressive_container_proof(
    field_roots: &[Hash256],
    active_fields: [u8; BYTES_PER_CHUNK],
    field_index: usize,
) -> Result<Vec<Hash256>, Error> {
    if field_roots.is_empty() {
        return Err(Error::NoFields);
    }
    if field_index >= field_roots.len() {
        return Err(Error::FieldIndexOutOfBounds {
            index: field_index,
            len: field_roots.len(),
        });
    }

    let (level, offset) = locate(field_index);
    let levels = level_roots(field_roots);

    let start = level_start(level);
    let end = (start + level_size(level)).min(field_roots.len());
    let tree = binary_tree(&field_roots[start..end], level_size(level));

    let mut branch = Vec::with_capacity(progressive_container_depth(field_index));

    // Sibling path up the binary subtree holding the field.
    let mut index = offset;
    for layer in tree.iter().take(tree.len().saturating_sub(1)) {
        branch.push(layer[index ^ 1]);
        index >>= 1;
    }

    // The rest of the spine hangs off the right of this level.
    branch.push(rest_root(&levels, level + 1));

    // Walking back up the spine, each ancestor's sibling is that level's binary subtree.
    for lower in (0..level).rev() {
        branch.push(levels[lower]);
    }

    // Finally the mixed in `active_fields`.
    branch.push(Hash256::from(active_fields));

    Ok(branch)
}

/// Verify `branch` proves `leaf` sits at `gindex` under `root`.
///
/// The bits of `gindex` above its leading one describe the path, so this handles the ragged depths
/// a progressive container produces without being told the depth separately.
pub fn is_valid_merkle_branch(
    leaf: Hash256,
    branch: &[Hash256],
    gindex: u64,
    root: Hash256,
) -> bool {
    if gindex == 0 {
        return false;
    }
    if branch.len() != gindex.ilog2() as usize {
        return false;
    }

    let mut value = leaf;
    let mut index = gindex;
    for sibling in branch {
        value = if index & 1 == 0 {
            Hash256::from(hash32_concat(value.as_slice(), sibling.as_slice()))
        } else {
            Hash256::from(hash32_concat(sibling.as_slice(), value.as_slice()))
        };
        index >>= 1;
    }

    value == root
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct, non-zero field roots so a misordered branch cannot pass by accident.
    fn field_roots(n: usize) -> Vec<Hash256> {
        (0..n)
            .map(|i| {
                let mut bytes = [0u8; 32];
                bytes[0..8].copy_from_slice(&((i as u64) + 1).to_le_bytes());
                Hash256::from(bytes)
            })
            .collect()
    }

    fn active_fields(n: usize) -> [u8; BYTES_PER_CHUNK] {
        let mut bytes = [0u8; BYTES_PER_CHUNK];
        for i in 0..n {
            bytes[i / 8] |= 1 << (i % 8);
        }
        bytes
    }

    /// The proof module rebuilds the tree that `ProgressiveMerkleHasher` streams. If the two ever
    /// disagree, proofs would verify against a root the hasher never produces, which is the one
    /// failure this module must not have.
    #[test]
    fn root_agrees_with_the_streaming_hasher() {
        for n in 1..=130 {
            let roots = field_roots(n);

            let mut hasher = crate::ProgressiveMerkleHasher::new();
            for root in &roots {
                hasher.write(root.as_slice()).unwrap();
            }
            let streamed = hasher.finish().unwrap();

            assert_eq!(
                progressive_root(&roots),
                streamed,
                "rebuilt tree disagrees with the hasher for {n} fields"
            );
        }
    }

    #[test]
    fn levels_start_where_the_spine_says() {
        assert_eq!(level_start(0), 0);
        assert_eq!(level_start(1), 1);
        assert_eq!(level_start(2), 5);
        assert_eq!(level_start(3), 21);
        assert_eq!(level_start(4), 85);
    }

    #[test]
    fn gindices_match_the_gloas_beacon_state() {
        // Checked against `eth-remerkleable`, the library the pyspec merkleizes with, for the 46
        // field Gloas `BeaconState`.
        assert_eq!(progressive_container_gindex(20), 367);
        assert_eq!(progressive_container_gindex(23), 2946);
        assert_eq!(progressive_container_gindex(24), 2947);

        // The depths are ragged, which the old fixed depth scheme assumed away.
        assert_eq!(progressive_container_depth(20), 8);
        assert_eq!(progressive_container_depth(23), 11);
        assert_eq!(progressive_container_depth(24), 11);
    }

    #[test]
    fn gindex_depth_agrees_with_branch_length() {
        for i in 0..200 {
            assert_eq!(
                progressive_container_gindex(i).ilog2() as usize,
                progressive_container_depth(i),
                "field {i}"
            );
        }
    }

    #[test]
    fn every_field_proves_against_the_root() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active);

        for i in 0..roots.len() {
            let branch = progressive_container_proof(&roots, active, i).unwrap();
            assert_eq!(branch.len(), progressive_container_depth(i), "field {i}");
            assert!(
                is_valid_merkle_branch(roots[i], &branch, progressive_container_gindex(i), root),
                "field {i} failed to verify"
            );
        }
    }

    #[test]
    fn proofs_hold_across_container_sizes() {
        for n in 1..=100 {
            let roots = field_roots(n);
            let active = active_fields(n);
            let root = progressive_container_root(&roots, active);

            for i in 0..n {
                let branch = progressive_container_proof(&roots, active, i).unwrap();
                assert!(
                    is_valid_merkle_branch(roots[i], &branch, progressive_container_gindex(i), root),
                    "container of {n} fields, field {i}"
                );
            }
        }
    }

    #[test]
    fn a_tampered_leaf_is_rejected() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active);

        let branch = progressive_container_proof(&roots, active, 24).unwrap();
        let gindex = progressive_container_gindex(24);

        assert!(is_valid_merkle_branch(roots[24], &branch, gindex, root));
        assert!(!is_valid_merkle_branch(roots[23], &branch, gindex, root));
    }

    #[test]
    fn a_tampered_branch_is_rejected() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active);
        let gindex = progressive_container_gindex(24);

        let mut branch = progressive_container_proof(&roots, active, 24).unwrap();
        branch[0] = Hash256::ZERO;
        assert!(!is_valid_merkle_branch(roots[24], &branch, gindex, root));

        // Swapping two siblings keeps the length but breaks the path.
        let mut branch = progressive_container_proof(&roots, active, 24).unwrap();
        branch.swap(0, 1);
        assert!(!is_valid_merkle_branch(roots[24], &branch, gindex, root));
    }

    #[test]
    fn a_branch_from_the_wrong_field_is_rejected() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active);

        // Field 23 and 24 are siblings at the same depth, so this is not caught by length alone.
        let branch = progressive_container_proof(&roots, active, 23).unwrap();
        assert!(!is_valid_merkle_branch(
            roots[24],
            &branch,
            progressive_container_gindex(24),
            root
        ));
    }

    #[test]
    fn active_fields_are_bound_to_the_root() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active);

        // A container claiming a different active field set must not verify against this root.
        let branch = progressive_container_proof(&roots, active_fields(45), 24).unwrap();
        assert!(!is_valid_merkle_branch(
            roots[24],
            &branch,
            progressive_container_gindex(24),
            root
        ));
    }

    #[test]
    fn out_of_bounds_fields_are_rejected() {
        let roots = field_roots(46);
        let active = active_fields(46);

        assert_eq!(
            progressive_container_proof(&roots, active, 46),
            Err(Error::FieldIndexOutOfBounds { index: 46, len: 46 })
        );
        assert_eq!(
            progressive_container_proof(&[], active, 0),
            Err(Error::NoFields)
        );
    }
}
