/*
 * Copyright 2019 The Starlark in Rust Authors.
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::fmt::Debug;

use allocative::Allocative;
use pagable::Pagable;
use pagable::PagableBoxDeserialize;
use pagable::PagableDeserialize;
use pagable::PagableDeserializer;
use pagable::PagableSerialize;
use pagable::PagableSerializer;
use pagable::pagable_typetag;
use pagable::typetag::PagableTagged;
use starlark_derive::type_matcher;

use crate as starlark;
use crate::typing::custom::TyCustom;
use crate::values::Value;
use crate::values::typing::type_compiled::alloc::TypeMatcherAlloc;
use crate::values::typing::type_compiled::type_matcher_factory::TypeMatcherFactory;

/// Marker trait for type matchers which are registered.
///
/// This trait is automatically implemented by the `#[type_matcher]` proc macro.
///
/// # Safety
///
/// This trait must only be implemented by the `#[type_matcher]` proc macro,
/// which ensures the type is properly registered in the vtable registry.
/// Manual implementations may break deserialization.
#[cfg_attr(not(feature = "pagable"), allow(dead_code))]
pub unsafe trait TypeMatcherRegistered {}

/// Base trait for type matchers
///
///  When `pagable` is enabled, matchers must also implement `TypeMatcherRegistered`
/// to ensure they are registered.
#[cfg(feature = "pagable")]
pub trait TypeMatcherBase:
    TypeMatcherRegistered + Allocative + Debug + Clone + Sized + Send + Sync + 'static
{
}

#[cfg(feature = "pagable")]
impl<T> TypeMatcherBase for T where
    T: TypeMatcherRegistered + Allocative + Debug + Clone + Sized + Send + Sync + 'static
{
}

/// Base trait for type matchers
#[cfg(not(feature = "pagable"))]
pub trait TypeMatcherBase: Allocative + Debug + Clone + Sized + Send + Sync + 'static {}

#[cfg(not(feature = "pagable"))]
impl<T> TypeMatcherBase for T where T: Allocative + Debug + Clone + Sized + Send + Sync + 'static {}

/// Runtime type matcher. E.g. when `isinstance(1, int)` is called,
/// implementation of `TypeMatcher` for `int` is used.
pub trait TypeMatcher: TypeMatcherBase + Pagable + PagableTagged {
    /// Check if the value matches the type.
    fn matches(&self, value: Value) -> bool;
    /// True if this matcher matches any value.
    fn is_wildcard(&self) -> bool {
        false
    }
}

#[pagable_typetag]
#[doc(hidden)]
pub trait TypeMatcherDyn: Debug + Allocative + PagableTagged + Send + Sync + 'static {
    fn matches_dyn(&self, value: Value) -> bool;
    fn is_wildcard_dyn(&self) -> bool;

    fn to_box(&self) -> TypeMatcherBox;
}

impl<T: TypeMatcher> TypeMatcherDyn for T {
    fn matches_dyn(&self, value: Value) -> bool {
        TypeMatcher::matches(self, value)
    }

    fn is_wildcard_dyn(&self) -> bool {
        TypeMatcher::is_wildcard(self)
    }

    fn to_box(&self) -> TypeMatcherBox {
        TypeMatcherBox::new(self.clone())
    }
}

#[pagable_typetag(TypeMatcherDyn)]
#[doc(hidden)]
#[derive(Debug, Allocative)]
pub struct TypeMatcherBox(pub(crate) Box<dyn TypeMatcherDyn>);

// Do NOT `#[derive(pagable::Pagable)]` here. The derive would generate
// asymmetric ser/de for the `Box<dyn TypeMatcherDyn>` field:
//   - Serialize: blanket `PagableSerialize for Box<T>` calls `(**self).pagable_serialize(…)`
//     on the concrete value — writes payload WITHOUT a type tag.
//   - Deserialize: blanket `PagableDeserialize for Box<T: PagableBoxDeserialize>` calls
//     `<dyn TypeMatcherDyn>::deserialize_box(…)` — reads a type tag then dispatches via
//     the typetag registry.
// The two halves don't agree on the wire format, so round-trips fail at runtime.
// Use `serialize_tagged` on the serialize side (writes tag + payload) to keep both
// ends symmetric.
impl PagableSerialize for TypeMatcherBox {
    fn pagable_serialize(&self, serializer: &mut dyn PagableSerializer) -> pagable::Result<()> {
        self.0.serialize_tagged(serializer)
    }
}

impl<'de> PagableDeserialize<'de> for TypeMatcherBox {
    fn pagable_deserialize<D: PagableDeserializer<'de> + ?Sized>(
        deserializer: &mut D,
    ) -> pagable::Result<Self> {
        Ok(TypeMatcherBox(<dyn TypeMatcherDyn>::deserialize_box(
            deserializer,
        )?))
    }
}

impl TypeMatcherBox {
    pub(crate) fn new<T: TypeMatcher>(matcher: T) -> TypeMatcherBox {
        TypeMatcherBox(Box::new(matcher))
    }
}

impl Clone for TypeMatcherBox {
    fn clone(&self) -> Self {
        self.0.to_box()
    }
}

#[type_matcher]
impl TypeMatcher for TypeMatcherBox {
    fn matches(&self, value: Value) -> bool {
        self.0.matches_dyn(value)
    }

    fn is_wildcard(&self) -> bool {
        self.0.is_wildcard_dyn()
    }
}

/// Type allocator which allocates `TypeMatcher` into `TypeMatcherBox`.
pub(crate) struct TypeMatcherBoxAlloc;

impl TypeMatcherAlloc for TypeMatcherBoxAlloc {
    type Result = TypeMatcherBox;

    fn alloc<T: TypeMatcher>(self, matcher: T) -> Self::Result {
        TypeMatcherBox::new(matcher)
    }

    fn custom(self, custom: &TyCustom) -> Self::Result {
        custom.matcher_with_box()
    }

    fn from_type_matcher_factory(self, factory: &TypeMatcherFactory) -> Self::Result {
        factory.factory.matcher_box()
    }
}
