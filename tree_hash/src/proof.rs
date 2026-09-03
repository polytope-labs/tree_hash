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

/// The field roots of a container, in chunk order.
pub type FieldRoots = Vec<Hash256>;

/// The field roots of a **balanced** container, which is what [`generate_multiproof`] accepts.
///
/// This is a distinct type from [`FieldRoots`] on purpose. A progressive container's roots
/// merkleize into the progressive spine, not a balanced tree, so handing them to the balanced
/// multiproof builder would produce a proof against a root the container never has. Only
/// [`ContainerFields::field_roots`] produces this type implicitly; building one by hand with
/// [`BalancedFieldRoots::new`] is a deliberate statement that the roots form a balanced tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BalancedFieldRoots(FieldRoots);

impl BalancedFieldRoots {
    /// Wrap roots that are known to merkleize as a balanced tree padded to a power of two.
    pub fn new(roots: FieldRoots) -> Self {
        Self(roots)
    }

    /// The roots in chunk order.
    pub fn as_slice(&self) -> &[Hash256] {
        &self.0
    }

    /// Unwrap the roots.
    pub fn into_inner(self) -> FieldRoots {
        self.0
    }
}

impl core::ops::Deref for BalancedFieldRoots {
    type Target = [Hash256];

    fn deref(&self) -> &[Hash256] {
        &self.0
    }
}

/// A **balanced** container that can hand out its field roots.
///
/// Derived by `#[derive(TreeHash)]` for ordinary containers only. Progressive containers get
/// [`TreeHashFields`] instead, deliberately: their field roots merkleize into the progressive
/// spine, so feeding them to [`generate_multiproof`], which builds a balanced tree, would yield a
/// proof against a root the container never produces. The two traits return different types,
/// [`BalancedFieldRoots`] here and plain [`FieldRoots`] there, and `generate_multiproof` accepts
/// only the former, so that mistake is a type error rather than a wrong root at runtime.
pub trait ContainerFields {
    /// The container's field roots in chunk order.
    fn field_roots(&self) -> BalancedFieldRoots;

    /// Build a balanced multiproof over this container's fields.
    fn prove_fields(&self, gindices: &[u64]) -> Result<Vec<Hash256>, Error> {
        generate_multiproof(&self.field_roots(), gindices)
    }
}

/// A progressive container that can hand out its field roots, so proofs can be built over it.
///
/// Derived automatically by `#[derive(TreeHash)]` with
/// `#[tree_hash(struct_behaviour = "progressive_container")]`. `tree_hash_root` consumes the field
/// roots as it streams them, so this trait is what makes them available a second time.
pub trait TreeHashFields {
    /// The packed `active_fields` bitmask mixed into the container root.
    const ACTIVE_FIELDS: [u8; BYTES_PER_CHUNK];

    /// The container's field roots in chunk order, with a zero root for every inactive field so
    /// that positions, and therefore generalized indices, stay stable across forks.
    fn field_roots(&self) -> FieldRoots;

    /// The container root, which must equal this type's `tree_hash_root`.
    fn container_root(&self) -> Result<Hash256, Error> {
        progressive_container_root(&self.field_roots(), Self::ACTIVE_FIELDS)
    }

    /// Prove the field at `field_index`, returning its root and the branch to the container root.
    ///
    /// Verify with [`is_valid_merkle_branch`] against
    /// [`progressive_container_gindex(field_index)`](progressive_container_gindex).
    fn prove_gindex(&self, gindex: u64) -> Result<(Hash256, Vec<Hash256>), Error> {
        let field_index =
            field_index_for_gindex(gindex).ok_or(Error::GindexOutOfTree { gindex })?;
        self.prove_field(field_index)
    }

    fn prove_field(&self, field_index: usize) -> Result<(Hash256, Vec<Hash256>), Error> {
        let roots = self.field_roots();
        let leaf = *roots
            .get(field_index)
            .ok_or(Error::FieldIndexOutOfBounds {
                index: field_index,
                len: roots.len(),
            })?;
        let branch = progressive_container_proof(&roots, Self::ACTIVE_FIELDS, field_index)?;
        Ok((leaf, branch))
    }
}

/// Errors returned when building a proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The requested field index is not present in the supplied field roots.
    FieldIndexOutOfBounds { index: usize, len: usize },
    /// No field roots were supplied, so there is nothing to prove.
    NoFields,
    /// A multiproof was given a different number of leaves and indices.
    LeafCountMismatch { leaves: usize, indices: usize },
    /// A multiproof carried a different number of nodes than its indices require.
    ProofCountMismatch { proof: usize, expected: usize },
    /// A multiproof did not contain enough nodes to reach the root.
    IncompleteProof,
    /// A generalized index falls outside the tree built from the supplied field roots.
    GindexOutOfTree { gindex: u64 },
    /// `active_fields` does not describe exactly the supplied field roots. Two containers whose
    /// roots differ only by trailing zero leaves inside one spine level would otherwise share a
    /// root, so the bitmask has to pin the field count.
    NonCanonicalActiveFields { highest_set: Option<usize>, fields: usize },
    /// A field that `active_fields` marks inactive carried a non-zero root. Inactive fields hash
    /// as zero, so anything else is not a state any container can produce.
    InactiveFieldNotZero { index: usize },
    /// A multiproof index set contained 0, which addresses no node.
    ZeroIndex,
    /// A multiproof index set contained the same index twice.
    DuplicateIndex { gindex: u64 },
    /// A multiproof index set contained a node and one of its descendants. The descendant's leaf
    /// would never be hashed into the root, so any value would pass for it.
    OverlappingIndices { ancestor: u64, descendant: u64 },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::FieldIndexOutOfBounds { index, len } => {
                write!(f, "field index {index} out of bounds for {len} fields")
            }
            Error::NoFields => write!(f, "no field roots supplied"),
            Error::LeafCountMismatch { leaves, indices } => {
                write!(f, "{leaves} leaves for {indices} indices")
            }
            Error::ProofCountMismatch { proof, expected } => {
                write!(f, "proof has {proof} nodes, expected {expected}")
            }
            Error::IncompleteProof => write!(f, "proof did not reach the root"),
            Error::GindexOutOfTree { gindex } => write!(f, "gindex {gindex} is outside the tree"),
            Error::NonCanonicalActiveFields { highest_set, fields } => {
                write!(f, "active_fields highest set bit {highest_set:?} does not match {fields} fields")
            }
            Error::InactiveFieldNotZero { index } => {
                write!(f, "field {index} is inactive but its root is not zero")
            }
            Error::ZeroIndex => write!(f, "index 0 addresses no node"),
            Error::DuplicateIndex { gindex } => write!(f, "index {gindex} appears twice"),
            Error::OverlappingIndices { ancestor, descendant } => {
                write!(f, "index {ancestor} is an ancestor of {descendant}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// The number of leaves in the binary subtree at `level`, or `None` once that no longer fits.
///
/// The ceiling is whatever `usize` can actually hold on the target rather than a constant derived
/// from `u64`: on a 32 bit target such as wasm32 the shift runs out at level 16, and a hardcoded
/// 64 bit bound would wrap there instead of refusing.
fn level_size(level: usize) -> Option<usize> {
    1usize.checked_shl(u32::try_from(level.checked_mul(2)?).ok()?)
}

/// The index of the first field held by `level`, or `None` once that no longer fits a `usize`.
///
/// Levels hold `1, 4, 16, 64, ...` fields, so the first index of level `k` is `(4^k - 1) / 3`.
fn level_start(level: usize) -> Option<usize> {
    Some((level_size(level)? - 1) / 3)
}

/// The level holding `field_index`, and the offset of the field within that level.
///
/// Returns `None` for an index no progressive container can hold. The bound is checked before the
/// arithmetic that would otherwise wrap, so this terminates for every input, including on 32 bit
/// targets where a wrapping shift would previously loop forever.
fn locate(field_index: usize) -> Option<(usize, usize)> {
    let mut level = 0;
    loop {
        match level_start(level + 1) {
            // The next level starts at or below this index, so keep walking up the spine.
            Some(next) if next <= field_index => level += 1,
            // Either the next level starts past this index, or there is no next level. Both mean
            // the index belongs to this one, if it fits; bailing out on the ceiling would make the
            // topmost level unreachable even though its gindices invert.
            _ => break,
        }
    }
    let start = level_start(level)?;
    let offset = field_index.checked_sub(start)?;
    if offset >= level_size(level)? {
        return None;
    }
    Some((level, offset))
}

/// The generalized index of the left (binary subtree) child at `level`, within the progressive
/// tree alone, before `active_fields` is mixed in.
///
/// The spine puts these at `2, 6, 14, 30, ...`, which is `2^(k + 2) - 2`.
fn level_gindex(level: usize) -> Option<u64> {
    1u64.checked_shl(u32::try_from(level.checked_add(2)?).ok()?)?.checked_sub(2)
}

/// The generalized index of `field_index` in a progressive container.
///
/// This is the index against the container's own root, so it already accounts for the
/// `active_fields` mix in. Note that unlike a fixed depth container, two fields of the same
/// container generally sit at different depths.
///
/// Returns `None` for an index no progressive container can hold, rather than wrapping or
/// panicking on the shift.
///
/// ```
/// # use tree_hash::proof::progressive_container_gindex;
/// // Fields 20, 23 and 24 of a 46 field container.
/// assert_eq!(progressive_container_gindex(20), Some(367));
/// assert_eq!(progressive_container_gindex(23), Some(2946));
/// assert_eq!(progressive_container_gindex(24), Some(2947));
/// ```
pub fn progressive_container_gindex(field_index: usize) -> Option<u64> {
    let (level, offset) = locate(field_index)?;

    // Position within the progressive tree, whose root is the left child of the final root.
    let within = level_gindex(level)?
        .checked_mul(level_size(level)? as u64)?
        .checked_add(offset as u64)?;

    // Graft that subtree under the left child of the final root. Stripping the leading one bit and
    // re-attaching it below the root's left child is the same as adding it back one place higher.
    within.checked_add(1u64 << within.ilog2())
}

/// The depth of `field_index`, which is the number of nodes in its branch.
///
/// Returns `None` for an index no progressive container can hold.
pub fn progressive_container_depth(field_index: usize) -> Option<usize> {
    // `2 * level` within the binary subtree, one for the spine sibling at this level, `level` more
    // walking back up the spine, and one for `active_fields`, which is `3 * level + 2`. It is read
    // off the gindex rather than recomputed so the two can never disagree about which indices fit.
    progressive_container_gindex(field_index).map(|gindex| gindex.ilog2() as usize)
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

/// The binary subtree for each level covered by `field_roots`, bottom up per level.
fn level_trees(field_roots: &[Hash256]) -> Vec<Vec<Vec<Hash256>>> {
    let mut trees = Vec::new();
    let mut level = 0;
    while let (Some(start), Some(size)) = (level_start(level), level_size(level)) {
        if start >= field_roots.len() {
            break;
        }
        let end = (start + size).min(field_roots.len());
        trees.push(binary_tree(&field_roots[start..end], size));
        level += 1;
    }
    trees
}

/// The root of a bottom up binary tree.
fn tree_root(tree: &[Vec<Hash256>]) -> Hash256 {
    *tree
        .last()
        .and_then(|layer| layer.first())
        .expect("tree has a root")
}

/// The binary subtree root for each level covered by `field_roots`.
fn level_roots(field_roots: &[Hash256]) -> Vec<Hash256> {
    level_trees(field_roots)
        .iter()
        .map(|tree| tree_root(tree))
        .collect()
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

/// The position of the highest set bit in `active_fields`, if any.
fn highest_active_bit(active_fields: &[u8; BYTES_PER_CHUNK]) -> Option<usize> {
    active_fields
        .iter()
        .enumerate()
        .rev()
        .find(|(_, byte)| **byte != 0)
        .map(|(index, byte)| index * 8 + (7 - byte.leading_zeros() as usize))
}

/// Check that `active_fields` describes exactly `field_roots`.
///
/// The spec's canonical form requires the bitmask's highest set bit to sit at the last field, and
/// every inactive field to hash as zero. The derive guarantees both at compile time, so this only
/// guards the raw API: without the first, two containers differing solely by trailing zero leaves
/// within a spine level would produce the same root, and the root would not commit to how many
/// fields were declared; without the second, a caller could pass a state no container produces.
fn check_active_fields(
    active_fields: &[u8; BYTES_PER_CHUNK],
    field_roots: &[Hash256],
) -> Result<(), Error> {
    let len = field_roots.len();
    let highest_set = highest_active_bit(active_fields);
    if highest_set != len.checked_sub(1) {
        return Err(Error::NonCanonicalActiveFields { highest_set, fields: len });
    }
    for (index, root) in field_roots.iter().enumerate() {
        let active = active_fields[index / 8] & (1 << (index % 8)) != 0;
        if !active && *root != Hash256::ZERO {
            return Err(Error::InactiveFieldNotZero { index });
        }
    }
    Ok(())
}

/// The full container root, with `active_fields` mixed in.
pub fn progressive_container_root(
    field_roots: &[Hash256],
    active_fields: [u8; BYTES_PER_CHUNK],
) -> Result<Hash256, Error> {
    check_active_fields(&active_fields, field_roots)?;
    Ok(crate::mix_in_active_fields(&progressive_root(field_roots), active_fields))
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
    check_active_fields(&active_fields, field_roots)?;

    let (level, offset) = locate(field_index).ok_or(Error::FieldIndexOutOfBounds {
        index: field_index,
        len: field_roots.len(),
    })?;

    // Every level's subtree is built once; the field's own level is read back out of the same set
    // rather than rebuilt.
    let trees = level_trees(field_roots);
    let levels: Vec<Hash256> = trees.iter().map(|tree| tree_root(tree)).collect();
    let tree = trees.get(level).ok_or(Error::FieldIndexOutOfBounds {
        index: field_index,
        len: field_roots.len(),
    })?;

    let mut branch = Vec::new();

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
///
/// # The caller owns the gindex
///
/// A proof of an interior node is a perfectly valid proof *at that node's gindex*. This function
/// answers only "does this branch put this leaf at this index", so the index must come from the
/// verifier's own configuration and never from the prover or from the proof being checked. Taking
/// it from the proof would let a prover choose which position it is proving, which is no proof at
/// all.
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

/// The field index a generalized index refers to, inverting [`progressive_container_gindex`].
///
/// Callers hold gindices in configuration (the beacon state's finalized root, next sync committee
/// and execution leaf are named that way), while proof building works in field indices.
pub fn field_index_for_gindex(gindex: u64) -> Option<usize> {
    if gindex < 2 {
        return None;
    }
    let top = gindex.ilog2();

    // The container hangs off the root's *left* child, so for any real field the bit just below
    // the leading one is clear. When it is set the gindex addresses the `active_fields` side of
    // the mix in; gindex 3 is that leaf, and without this check it would be mistaken for field 0.
    if top >= 1 && (gindex >> (top - 1)) & 1 == 1 {
        return None;
    }

    // Undo the graft under the root's left child.
    let within = gindex.checked_sub(1u64 << top.checked_sub(1)?)?;

    let mut level = 0;
    loop {
        let base = level_gindex(level)?.checked_mul(level_size(level)? as u64)?;
        if within < base {
            return None;
        }
        if within < base.checked_add(level_size(level)? as u64)? {
            return Some(level_start(level)? + (within - base) as usize);
        }
        level += 1;
    }
}

/// Build a multiproof over a balanced container's field roots.
///
/// The pre-Gloas execution payload header is an ordinary container, and its state root, block
/// number and timestamp are proven together, so this is the generating counterpart to
/// [`multiproof::calculate_multi_merkle_root`] and [`multiproof::verify_merkle_multiproof`].
///
/// Takes [`BalancedFieldRoots`] rather than a bare slice so a progressive container's roots, which
/// do not merkleize as a balanced tree, cannot be passed here by mistake.
pub fn generate_multiproof(
    field_roots: &BalancedFieldRoots,
    gindices: &[u64],
) -> Result<Vec<Hash256>, Error> {
    let field_roots = field_roots.as_slice();
    if field_roots.is_empty() {
        return Err(Error::NoFields);
    }
    multiproof::validate_indices(gindices)?;
    let leaves = field_roots.len().next_power_of_two();
    let tree = binary_tree(field_roots, leaves);

    multiproof::get_helper_indices(gindices)
        .into_iter()
        .map(|gindex| node_at(&tree, gindex).ok_or(Error::GindexOutOfTree { gindex }))
        .collect()
}

/// Read the node at `gindex` from a bottom up binary tree.
fn node_at(tree: &[Vec<Hash256>], gindex: u64) -> Option<Hash256> {
    let depth = gindex.ilog2() as usize;
    let total = tree.len().checked_sub(1)?;
    let layer = tree.get(total.checked_sub(depth)?)?;
    layer.get((gindex - (1 << depth)) as usize).copied()
}

/// Multiproofs over ordinary fixed depth containers.
///
/// Progressive containers cover the Gloas `BeaconState`, but pre-Gloas forks still prove several
/// fields of one container at once (the execution payload header's state root, block number and
/// timestamp), and those containers are ordinary balanced trees. This is the multiproof algorithm
/// from the SSZ specification (`ethereum/ssz-specs`, formerly `ssz/merkle-proofs.md` in
/// `ethereum/consensus-specs`), which both shapes share.
///
/// Verifiers should call [`verify_merkle_multiproof`], which compares against the expected root.
/// [`calculate_multi_merkle_root`] only recomputes a root, and a tampered leaf is never
/// structurally invalid, so an `Ok` from it says nothing about whether the leaves are genuine.
pub mod multiproof {
    use super::*;
    use alloc::collections::{BTreeMap, BTreeSet};

    /// The sibling of `index`.
    const fn sibling(index: u64) -> u64 {
        index ^ 1
    }

    /// The generalized indices of the siblings on the path from `index` up to the root.
    fn branch_indices(index: u64) -> Vec<u64> {
        let mut out = Vec::new();
        let mut index = index;
        while index > 1 {
            out.push(sibling(index));
            index /= 2;
        }
        out
    }

    /// The generalized indices on the path from `index` up to, but excluding, the root.
    fn path_indices(index: u64) -> Vec<u64> {
        let mut out = Vec::new();
        let mut index = index;
        while index > 1 {
            out.push(index);
            index /= 2;
        }
        out
    }

    /// Reject index sets that would leave a leaf unverified.
    ///
    /// The algorithm in the consensus specs combines a node with its sibling and stops once a
    /// parent is already known. If one index is an ancestor of another, the descendant's leaf is
    /// never hashed into the root, so **any** value passes for it. The spec gets away with this
    /// because it is always driven by hardcoded index sets; a runtime that takes indices from
    /// configuration or untrusted data does not have that guarantee, so the shape is checked here.
    ///
    /// Linearithmic rather than pairwise: duplicates fall out of a sort, and an ancestor is found
    /// by walking each index up its own path, which is at most 64 steps, against the set.
    pub fn validate_indices(indices: &[u64]) -> Result<(), Error> {
        if indices.contains(&0) {
            return Err(Error::ZeroIndex);
        }

        let mut sorted: Vec<u64> = indices.to_vec();
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            if pair[0] == pair[1] {
                return Err(Error::DuplicateIndex { gindex: pair[0] });
            }
        }

        let seen: BTreeSet<u64> = sorted.iter().copied().collect();
        for &index in &sorted {
            // Every strict ancestor of `index` is one of `index >> 1`, `index >> 2`, ... down to
            // the root, so the path is walked instead of comparing against every other index.
            let mut ancestor = index >> 1;
            while ancestor > 0 {
                if seen.contains(&ancestor) {
                    return Err(Error::OverlappingIndices { ancestor, descendant: index });
                }
                ancestor >>= 1;
            }
        }
        Ok(())
    }

    /// The generalized indices whose roots a verifier must be given to recompute the root from
    /// `indices`, ordered deepest first.
    ///
    /// These are every sibling along every path, minus the nodes the proof can already derive.
    pub fn get_helper_indices(indices: &[u64]) -> Vec<u64> {
        let mut helpers = BTreeSet::new();
        let mut known = BTreeSet::new();

        for index in indices {
            helpers.extend(branch_indices(*index));
            known.extend(path_indices(*index));
            known.insert(*index);
        }

        let mut out: Vec<u64> = helpers.difference(&known).copied().collect();
        // Deepest first, which is the order `calculate_multi_merkle_root` consumes them in.
        out.sort_unstable_by(|a, b| b.cmp(a));
        out
    }

    /// Recompute the root from `leaves` at `indices`, given the `proof` nodes named by
    /// [`get_helper_indices`] in the same order.
    ///
    /// This is the prover side and the building block. An `Ok` here does **not** mean the leaves
    /// are genuine, only that the proof has the right shape; a tampered leaf simply produces a
    /// different root. Verifiers must compare the result against a root they already trust, which
    /// is what [`verify_merkle_multiproof`] does.
    pub fn calculate_multi_merkle_root(
        leaves: &[Hash256],
        proof: &[Hash256],
        indices: &[u64],
    ) -> Result<Hash256, Error> {
        if leaves.len() != indices.len() {
            return Err(Error::LeafCountMismatch {
                leaves: leaves.len(),
                indices: indices.len(),
            });
        }
        validate_indices(indices)?;
        let helpers = get_helper_indices(indices);
        if proof.len() != helpers.len() {
            return Err(Error::ProofCountMismatch {
                proof: proof.len(),
                expected: helpers.len(),
            });
        }

        let mut objects: BTreeMap<u64, Hash256> = indices
            .iter()
            .copied()
            .zip(leaves.iter().copied())
            .chain(helpers.iter().copied().zip(proof.iter().copied()))
            .collect();

        // Walk deepest first, combining any pair whose parent is not yet known.
        let mut keys: Vec<u64> = objects.keys().copied().collect();
        keys.sort_unstable_by(|a, b| b.cmp(a));

        let mut pos = 0;
        while pos < keys.len() {
            let key = keys[pos];
            let has_sibling = objects.contains_key(&sibling(key));
            let parent = key / 2;
            if key > 1 && has_sibling && !objects.contains_key(&parent) {
                let left = objects[&(key & !1)];
                let right = objects[&(key | 1)];
                objects.insert(
                    parent,
                    Hash256::from(hash32_concat(left.as_slice(), right.as_slice())),
                );
                keys.push(parent);
            }
            pos += 1;
        }

        objects.get(&1).copied().ok_or(Error::IncompleteProof)
    }

    /// Verify that `proof` places `leaves` at `indices` under `root`.
    ///
    /// The verifier side of [`calculate_multi_merkle_root`]: the recomputed root is compared to
    /// the expected one here, so the caller cannot mistake a well formed proof of the wrong values
    /// for a valid one. As with [`is_valid_merkle_branch`](super::is_valid_merkle_branch), the
    /// indices must come from the verifier's own configuration, never from the prover.
    pub fn verify_merkle_multiproof(
        leaves: &[Hash256],
        proof: &[Hash256],
        indices: &[u64],
        root: Hash256,
    ) -> bool {
        calculate_multi_merkle_root(leaves, proof, indices) == Ok(root)
    }
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

    /// Reported in review: if one index is an ancestor of another, the descendant's leaf is never
    /// hashed into the root, so any value passes for it. Not reachable through the light client's
    /// hardcoded same depth indices, but it must not be reachable at all.
    #[test]
    fn overlapping_indices_are_rejected() {
        use crate::proof::multiproof::calculate_multi_merkle_root;

        let leaf = Hash256::repeat_byte(1);
        let garbage = Hash256::repeat_byte(0xff);

        // ancestor / descendant
        assert_eq!(
            calculate_multi_merkle_root(&[leaf, garbage], &[], &[2, 5]),
            Err(Error::OverlappingIndices { ancestor: 2, descendant: 5 })
        );
        // the root itself as an index
        assert_eq!(
            calculate_multi_merkle_root(&[leaf, garbage], &[], &[1, 4]),
            Err(Error::OverlappingIndices { ancestor: 1, descendant: 4 })
        );
        // duplicates
        assert_eq!(
            calculate_multi_merkle_root(&[garbage, leaf], &[], &[4, 4]),
            Err(Error::DuplicateIndex { gindex: 4 })
        );
        // index 0 addresses no node
        assert_eq!(
            calculate_multi_merkle_root(&[garbage, leaf], &[], &[0, 4]),
            Err(Error::ZeroIndex)
        );
    }

    /// The ceiling must follow the target's pointer width, not a constant derived from `u64`.
    /// On wasm32 the shift runs out at level 16; on a 64 bit host at level 32. Checking it against
    /// `usize::BITS` makes the property hold wherever this is compiled, including the runtime.
    #[test]
    fn level_size_stops_at_the_target_width() {
        let last = (usize::BITS / 2) as usize;
        assert!(level_size(last - 1).is_some(), "level {} should fit", last - 1);
        assert_eq!(level_size(last), None, "level {last} should not fit");
        assert_eq!(level_size(usize::MAX), None);
    }

    /// Reported in the second review: the zero check ran per element, but `is_ancestor_of` takes a
    /// logarithm of both arguments, so a zero anywhere after the first index panicked before its
    /// own turn came round. The original test only covered the passing order.
    #[test]
    fn a_zero_index_is_rejected_in_any_position() {
        use crate::proof::multiproof::{calculate_multi_merkle_root, validate_indices};

        for indices in [vec![0, 4], vec![4, 0], vec![4, 0, 8], vec![4, 8, 0], vec![0]] {
            assert_eq!(validate_indices(&indices), Err(Error::ZeroIndex), "{indices:?}");
        }

        let leaf = Hash256::repeat_byte(1);
        assert_eq!(
            calculate_multi_merkle_root(&[leaf, leaf], &[], &[4, 0]),
            Err(Error::ZeroIndex)
        );
        assert_eq!(
            generate_multiproof(&BalancedFieldRoots::new(vec![leaf]), &[4, 0]),
            Err(Error::ZeroIndex)
        );
    }

    /// Indices at mixed depths that do not overlap must still be accepted.
    #[test]
    fn mixed_depth_indices_are_accepted() {
        use crate::proof::multiproof::validate_indices;
        assert_eq!(validate_indices(&[2, 6]), Ok(()));
        assert_eq!(validate_indices(&[4, 6]), Ok(()));
    }

    /// Reported in review: the deepest level that fits both a `usize` field index and a `u64`
    /// gindex must be reachable from both directions, and the level past it refused by both. On a
    /// 64 bit host that is level 20 (the `u64` gindex runs out first); on wasm32 it is level 15.
    #[test]
    fn the_top_level_round_trips() {
        let mut level = 0;
        while level_start(level + 1)
            .and_then(progressive_container_gindex)
            .is_some()
        {
            level += 1;
        }
        let first = level_start(level).expect("the top level has a start");
        let last = level_start(level + 1)
            .map(|next| next - 1)
            .unwrap_or_else(|| first + level_size(level).unwrap() - 1);
        for field in [first, last] {
            let gindex = progressive_container_gindex(field).expect("top level fits");
            assert_eq!(field_index_for_gindex(gindex), Some(field), "level {level}");
            assert_eq!(progressive_container_depth(field), Some(3 * level + 2));
        }
        // The level above is refused from both directions, and depth agrees with gindex there too.
        if let Some(next) = level_start(level + 1) {
            assert_eq!(progressive_container_gindex(next), None);
            assert_eq!(progressive_container_depth(next), None);
        }
        assert!(level >= 15, "at least level 15 must fit on every supported target");
    }

    /// Reported in review: `check_active_fields` pinned the highest bit but not that inactive
    /// positions actually hold zero roots.
    #[test]
    fn a_non_zero_root_at_an_inactive_field_is_rejected() {
        let mut roots = field_roots(6);
        let mut active = active_fields(6);
        active[0] &= !(1 << 2);

        // Zero at the inactive position is the canonical state and is accepted.
        roots[2] = Hash256::ZERO;
        assert!(progressive_container_root(&roots, active).is_ok());
        assert!(progressive_container_proof(&roots, active, 3).is_ok());

        roots[2] = Hash256::repeat_byte(0xaa);
        assert_eq!(
            progressive_container_root(&roots, active),
            Err(Error::InactiveFieldNotZero { index: 2 })
        );
        assert_eq!(
            progressive_container_proof(&roots, active, 3),
            Err(Error::InactiveFieldNotZero { index: 2 })
        );
    }

    /// Reported in review: `calculate_multi_merkle_root` returns `Ok` for a tampered leaf, since
    /// tampering is never structurally invalid. The verifier entry point compares the root.
    #[test]
    fn verify_merkle_multiproof_compares_the_root() {
        use crate::proof::multiproof::{calculate_multi_merkle_root, verify_merkle_multiproof};

        let roots = field_roots(8);
        let balanced = BalancedFieldRoots::new(roots.clone());
        let root = tree_root(&binary_tree(&roots, 8));
        let indices = [8 + 1, 8 + 6];
        let proof = generate_multiproof(&balanced, &indices).unwrap();

        assert!(verify_merkle_multiproof(
            &[roots[1], roots[6]],
            &proof,
            &indices,
            root
        ));

        // A tampered leaf still yields `Ok` from the calculator, and `false` from the verifier.
        let tampered = [roots[2], roots[6]];
        assert!(calculate_multi_merkle_root(&tampered, &proof, &indices).is_ok());
        assert!(!verify_merkle_multiproof(&tampered, &proof, &indices, root));

        // Malformed input is `false`, not a panic.
        assert!(!verify_merkle_multiproof(
            &[roots[1]],
            &proof,
            &indices,
            root
        ));
        assert!(!verify_merkle_multiproof(
            &[roots[1], roots[6]],
            &proof[1..],
            &indices,
            root
        ));
    }

    /// Sibling indices at the same depth, which is what the light client actually uses, still work.
    #[test]
    fn sibling_indices_are_still_accepted() {
        use crate::proof::multiproof::{calculate_multi_merkle_root, get_helper_indices};

        let roots = field_roots(16);
        let tree = binary_tree(&roots, 16);
        let root = *tree.last().unwrap().first().unwrap();
        let indices = [16 + 2, 16 + 8, 16 + 9];

        let proof = generate_multiproof(&BalancedFieldRoots::new(roots.clone()), &indices).unwrap();
        assert_eq!(proof.len(), get_helper_indices(&indices).len());
        assert_eq!(
            calculate_multi_merkle_root(&[roots[2], roots[8], roots[9]], &proof, &indices).unwrap(),
            root
        );
    }

    /// Reported in review: gindex 3 is the `active_fields` leaf, not field 0. The un-grafting step
    /// has to check the bit below the leading one.
    #[test]
    fn the_active_fields_leaf_is_not_a_field() {
        assert_eq!(field_index_for_gindex(3), None);
        assert_eq!(field_index_for_gindex(2), None);
        // and the real fields still invert
        assert_eq!(field_index_for_gindex(367), Some(20));
        assert_eq!(field_index_for_gindex(2947), Some(24));
    }

    /// Reported in review: unchecked shifts and multiplies panicked on 64 bit and looped forever on
    /// 32 bit. Out of range indices must simply be refused.
    #[test]
    fn out_of_range_indices_do_not_panic_or_hang() {
        assert_eq!(progressive_container_gindex(usize::MAX), None);
        assert_eq!(progressive_container_depth(usize::MAX), None);
        assert_eq!(field_index_for_gindex(u64::MAX), None);
        assert_eq!(field_index_for_gindex(1u64 << 62), None);

        // the boundary either resolves or refuses, but never wraps
        for i in [357_913_941usize, 1 << 40, usize::MAX / 2] {
            let _ = progressive_container_gindex(i);
            let _ = progressive_container_depth(i);
        }
    }

    #[test]
    fn levels_start_where_the_spine_says() {
        assert_eq!(level_start(0), Some(0));
        assert_eq!(level_start(1), Some(1));
        assert_eq!(level_start(2), Some(5));
        assert_eq!(level_start(3), Some(21));
        assert_eq!(level_start(4), Some(85));
    }

    #[test]
    fn gindices_match_the_gloas_beacon_state() {
        // Checked against `eth-remerkleable`, the library the pyspec merkleizes with, for the 46
        // field Gloas `BeaconState`.
        assert_eq!(progressive_container_gindex(20), Some(367));
        assert_eq!(progressive_container_gindex(23), Some(2946));
        assert_eq!(progressive_container_gindex(24), Some(2947));

        // The depths are ragged, which the old fixed depth scheme assumed away.
        assert_eq!(progressive_container_depth(20), Some(8));
        assert_eq!(progressive_container_depth(23), Some(11));
        assert_eq!(progressive_container_depth(24), Some(11));
    }

    #[test]
    fn gindex_depth_agrees_with_branch_length() {
        for i in 0..200 {
            assert_eq!(
                progressive_container_gindex(i).unwrap().ilog2() as usize,
                progressive_container_depth(i).unwrap(),
                "field {i}"
            );
        }
    }

    #[test]
    fn every_field_proves_against_the_root() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active).unwrap();

        for i in 0..roots.len() {
            let branch = progressive_container_proof(&roots, active, i).unwrap();
            assert_eq!(branch.len(), progressive_container_depth(i).unwrap(), "field {i}");
            assert!(
                is_valid_merkle_branch(roots[i], &branch, progressive_container_gindex(i).unwrap(), root),
                "field {i} failed to verify"
            );
        }
    }

    #[test]
    fn proofs_hold_across_container_sizes() {
        for n in 1..=100 {
            let roots = field_roots(n);
            let active = active_fields(n);
            let root = progressive_container_root(&roots, active).unwrap();

            for i in 0..n {
                let branch = progressive_container_proof(&roots, active, i).unwrap();
                assert!(
                    is_valid_merkle_branch(roots[i], &branch, progressive_container_gindex(i).unwrap(), root),
                    "container of {n} fields, field {i}"
                );
            }
        }
    }

    #[test]
    fn a_tampered_leaf_is_rejected() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active).unwrap();

        let branch = progressive_container_proof(&roots, active, 24).unwrap();
        let gindex = progressive_container_gindex(24).unwrap();

        assert!(is_valid_merkle_branch(roots[24], &branch, gindex, root));
        assert!(!is_valid_merkle_branch(roots[23], &branch, gindex, root));
    }

    #[test]
    fn a_tampered_branch_is_rejected() {
        let roots = field_roots(46);
        let active = active_fields(46);
        let root = progressive_container_root(&roots, active).unwrap();
        let gindex = progressive_container_gindex(24).unwrap();

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
        let root = progressive_container_root(&roots, active).unwrap();

        // Field 23 and 24 are siblings at the same depth, so this is not caught by length alone.
        let branch = progressive_container_proof(&roots, active, 23).unwrap();
        assert!(!is_valid_merkle_branch(
            roots[24],
            &branch,
            progressive_container_gindex(24).unwrap(),
            root
        ));
    }

    #[test]
    fn active_fields_are_bound_to_the_root() {
        let roots = field_roots(46);

        // Reported in review: without a canonical form check, two containers differing only by
        // trailing zero leaves inside one spine level share a root, so the root does not commit to
        // how many fields were declared. The mismatch is now refused rather than merely failing to
        // verify later, which is the stronger guarantee.
        assert_eq!(
            progressive_container_proof(&roots, active_fields(45), 24),
            Err(Error::NonCanonicalActiveFields { highest_set: Some(44), fields: 46 })
        );
        assert_eq!(
            progressive_container_root(&roots, active_fields(45)),
            Err(Error::NonCanonicalActiveFields { highest_set: Some(44), fields: 46 })
        );
        // An over-long mask is refused from the other side too.
        assert!(progressive_container_root(&roots, active_fields(47)).is_err());
        // The matching mask is accepted.
        assert!(progressive_container_root(&roots, active_fields(46)).is_ok());
    }

    #[test]
    fn gindex_inversion_round_trips() {
        for i in 0..200 {
            let gindex = progressive_container_gindex(i).unwrap();
            assert_eq!(field_index_for_gindex(gindex), Some(i), "field {i}");
        }
        // The gloas beacon state fields we care about.
        assert_eq!(field_index_for_gindex(367), Some(20));
        assert_eq!(field_index_for_gindex(2946), Some(23));
        assert_eq!(field_index_for_gindex(2947), Some(24));
        assert_eq!(field_index_for_gindex(0), None);
        assert_eq!(field_index_for_gindex(1), None);
    }

    /// Generation and verification must agree, or the prover emits proofs the verifier rejects.
    #[test]
    fn generated_multiproofs_verify() {
        use crate::proof::multiproof::calculate_multi_merkle_root;

        let roots = field_roots(16);
        let tree = binary_tree(&roots, 16);
        let root = *tree.last().unwrap().first().unwrap();

        // The pre-Gloas execution payload case: three fields of one container.
        let indices = [16 + 2, 16 + 8, 16 + 9];
        let picked = [roots[2], roots[8], roots[9]];

        let proof = generate_multiproof(&BalancedFieldRoots::new(roots.clone()), &indices).unwrap();
        assert_eq!(
            calculate_multi_merkle_root(&picked, &proof, &indices).unwrap(),
            root
        );
    }

    #[test]
    fn generated_multiproofs_verify_across_subsets() {
        use crate::proof::multiproof::calculate_multi_merkle_root;

        let roots = field_roots(16);
        let tree = binary_tree(&roots, 16);
        let root = *tree.last().unwrap().first().unwrap();

        for mask in 1u32..(1 << 8) {
            let picked: Vec<usize> = (0..8).filter(|i| mask & (1 << i) != 0).collect();
            let indices: Vec<u64> = picked.iter().map(|i| 16 + *i as u64).collect();
            let values: Vec<Hash256> = picked.iter().map(|i| roots[*i]).collect();

            let proof = generate_multiproof(&BalancedFieldRoots::new(roots.clone()), &indices).unwrap();
            assert_eq!(
                calculate_multi_merkle_root(&values, &proof, &indices).unwrap(),
                root,
                "subset {mask:08b}"
            );
        }
    }

    mod multiproofs {
        use super::*;
        use crate::proof::multiproof::{calculate_multi_merkle_root, get_helper_indices};

        /// Read the node at `gindex` out of a bottom up binary tree.
        fn node_at(tree: &[Vec<Hash256>], gindex: u64) -> Hash256 {
            let depth = gindex.ilog2() as usize;
            let total = tree.len() - 1;
            tree[total - depth][(gindex - (1 << depth)) as usize]
        }

        /// A balanced 16 leaf container, the shape of an ordinary fixed depth SSZ container.
        fn fixture() -> (Vec<Hash256>, Vec<Vec<Hash256>>, Hash256) {
            let leaves = field_roots(16);
            let tree = binary_tree(&leaves, 16);
            let root = *tree.last().unwrap().first().unwrap();
            (leaves, tree, root)
        }

        #[test]
        fn a_single_index_recomputes_the_root() {
            let (leaves, tree, root) = fixture();
            let index = 16 + 5;

            let helpers = get_helper_indices(&[index]);
            let proof: Vec<Hash256> = helpers.iter().map(|g| node_at(&tree, *g)).collect();

            assert_eq!(
                calculate_multi_merkle_root(&[leaves[5]], &proof, &[index]).unwrap(),
                root
            );
        }

        /// The pre-Gloas execution payload case: three fields of one container proven together.
        #[test]
        fn three_indices_recompute_the_root() {
            let (leaves, tree, root) = fixture();
            let indices = [16 + 2, 16 + 8, 16 + 9];
            let picked = [leaves[2], leaves[8], leaves[9]];

            let helpers = get_helper_indices(&indices);
            let proof: Vec<Hash256> = helpers.iter().map(|g| node_at(&tree, *g)).collect();

            assert_eq!(
                calculate_multi_merkle_root(&picked, &proof, &indices).unwrap(),
                root
            );
        }

        /// Sibling leaves share a parent, so the shared node must not be sent twice.
        #[test]
        fn sibling_indices_do_not_duplicate_helpers() {
            let (leaves, tree, root) = fixture();
            let indices = [16 + 8, 16 + 9];

            let helpers = get_helper_indices(&indices);
            assert!(!helpers.contains(&(16 + 8)));
            assert!(!helpers.contains(&(16 + 9)));

            let proof: Vec<Hash256> = helpers.iter().map(|g| node_at(&tree, *g)).collect();
            assert_eq!(
                calculate_multi_merkle_root(&[leaves[8], leaves[9]], &proof, &indices).unwrap(),
                root
            );
        }

        #[test]
        fn every_index_subset_recomputes_the_root() {
            let (leaves, tree, root) = fixture();

            // Exhaustive over all non-empty subsets of a 4 leaf prefix, plus a wider sweep.
            for mask in 1u32..(1 << 8) {
                let picked: Vec<usize> = (0..8).filter(|i| mask & (1 << i) != 0).collect();
                let indices: Vec<u64> = picked.iter().map(|i| 16 + *i as u64).collect();
                let values: Vec<Hash256> = picked.iter().map(|i| leaves[*i]).collect();

                let helpers = get_helper_indices(&indices);
                let proof: Vec<Hash256> = helpers.iter().map(|g| node_at(&tree, *g)).collect();

                assert_eq!(
                    calculate_multi_merkle_root(&values, &proof, &indices).unwrap(),
                    root,
                    "subset {mask:08b}"
                );
            }
        }

        #[test]
        fn a_tampered_leaf_does_not_reach_the_root() {
            let (leaves, tree, root) = fixture();
            let indices = [16 + 2, 16 + 8];

            let helpers = get_helper_indices(&indices);
            let proof: Vec<Hash256> = helpers.iter().map(|g| node_at(&tree, *g)).collect();

            let wrong = calculate_multi_merkle_root(&[leaves[3], leaves[8]], &proof, &indices)
                .unwrap();
            assert_ne!(wrong, root);
        }

        #[test]
        fn mismatched_lengths_are_rejected() {
            let (leaves, tree, _) = fixture();
            let indices = [16 + 2, 16 + 8];
            let helpers = get_helper_indices(&indices);
            let proof: Vec<Hash256> = helpers.iter().map(|g| node_at(&tree, *g)).collect();

            assert_eq!(
                calculate_multi_merkle_root(&[leaves[2]], &proof, &indices),
                Err(Error::LeafCountMismatch { leaves: 1, indices: 2 })
            );
            assert_eq!(
                calculate_multi_merkle_root(&[leaves[2], leaves[8]], &proof[1..], &indices),
                Err(Error::ProofCountMismatch {
                    proof: proof.len() - 1,
                    expected: proof.len()
                })
            );
        }
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
