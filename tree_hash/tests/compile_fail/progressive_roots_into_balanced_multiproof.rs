//! A progressive container's field roots merkleize into the progressive spine, so they must not be
//! accepted by the balanced multiproof builder. `TreeHashFields::field_roots` returns plain
//! `FieldRoots`, while `generate_multiproof` takes `BalancedFieldRoots`, which only
//! `ContainerFields` produces, so this is a type error rather than a proof against the wrong root.
use tree_hash::proof::{generate_multiproof, TreeHashFields};
use tree_hash_derive::TreeHash;

#[derive(TreeHash)]
#[tree_hash(struct_behaviour = "progressive_container", active_fields(1, 1))]
struct Foo {
    a: u8,
    b: u8,
}

fn main() {
    let foo = Foo { a: 1, b: 2 };
    let _ = generate_multiproof(&foo.field_roots(), &[2]);
}
