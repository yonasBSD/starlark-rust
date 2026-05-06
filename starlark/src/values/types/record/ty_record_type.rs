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

use allocative::Allocative;
use starlark_derive::StarlarkPagable;

use crate as starlark;
use crate::eval::ParametersSpec;
use crate::typing::Ty;
use crate::values::FrozenValue;

#[derive(Allocative, Debug, StarlarkPagable)]
#[doc(hidden)]
pub struct TyRecordData {
    /// Name of the record type.
    pub(crate) name: String,
    /// Type of record instance.
    #[starlark_pagable(pagable)]
    pub(crate) ty_record: Ty,
    /// Type of record type.
    #[starlark_pagable(pagable)]
    pub(crate) ty_record_type: Ty,
    /// Creating these on every invoke is pretty expensive (profiling shows)
    /// so compute them in advance and cache.
    pub(crate) parameter_spec: ParametersSpec<FrozenValue>,
}

// `pagable::Pagable` bridge for `TyRecordData`. Lets `Arc<TyRecordData>` use
// pagable's Arc-identity dedup mechanism
impl pagable::PagableSerialize for TyRecordData {
    fn pagable_serialize(
        &self,
        serializer: &mut dyn pagable::PagableSerializer,
    ) -> pagable::Result<()> {
        let mut ctx = crate::pagable::StarlarkSerializerImpl::recover_from_pagable(serializer)
            .map_err(|e: crate::Error| e.into_anyhow())?;
        <Self as crate::pagable::StarlarkSerialize>::starlark_serialize(self, &mut ctx)
            .map_err(|e: crate::Error| e.into_anyhow())
    }
}

impl<'de> pagable::PagableDeserialize<'de> for TyRecordData {
    fn pagable_deserialize<D: pagable::PagableDeserializer<'de> + ?Sized>(
        deserializer: &mut D,
    ) -> pagable::Result<Self> {
        let mut ctx =
            crate::pagable::StarlarkDeserializerImpl::recover_from_pagable(deserializer.as_dyn())
                .map_err(|e: crate::Error| e.into_anyhow())?;
        <Self as crate::pagable::StarlarkDeserialize>::starlark_deserialize(&mut ctx)
            .map_err(|e: crate::Error| e.into_anyhow())
    }
}

#[cfg(test)]
mod tests {
    use crate::assert;

    #[test]
    fn test_good() {
        assert::pass(
            r#"
MyRec = record(x = int)

def foo(x: MyRec): pass

foo(MyRec(x = 1))
        "#,
        );
    }

    #[test]
    fn test_fail_compile_time() {
        assert::fail_golden(
            "src/values/types/record/ty_record_type/fail_compile_time.golden",
            r#"
MyRec = record(x = int)
WrongRec = record(x = int)

def foo(x: MyRec): pass

def bar():
    foo(WrongRec(x = 1))
        "#,
        );
    }

    #[test]
    fn test_fail_runtime_time() {
        assert::fail_golden(
            "src/values/types/record/ty_record_type/fail_runtime_time.golden",
            r#"
MyRec = record(x = int)
WrongRec = record(x = int)

def foo(x: MyRec): pass

noop(foo)(WrongRec(x = 1))
        "#,
        );
    }

    #[test]
    fn test_record_instance_typechecker_ty() {
        assert::pass(
            r#"
MyRec = record(x = int)
X = MyRec(x = 1)

def foo() -> MyRec:
    # This fails if record instance does not override `typechecker_ty`.
    return X
"#,
        );
    }

    #[test]
    fn test_typecheck_field_pass() {
        assert::pass(
            r#"
MyRec = record(x = int, y = int)

def f(rec: MyRec) -> int:
    return rec.x + rec.y

assert_eq(f(MyRec(x = 1, y = 2)), 3)
"#,
        );
    }

    #[test]
    fn test_typecheck_field_fail() {
        assert::fail_golden(
            "src/values/types/record/ty_record_type/typecheck_field_fail.golden",
            r#"
MyRec = record(x = int, y = int)

def f(rec: MyRec) -> int:
    return rec.z
"#,
        );
    }

    #[test]
    fn test_typecheck_record_type_call() {
        assert::fail_golden(
            "src/values/types/record/ty_record_type/typecheck_record_type_call.golden",
            r#"
MyRec = record(x = int)

def test():
    MyRec(x = "")
"#,
        );
    }
}
