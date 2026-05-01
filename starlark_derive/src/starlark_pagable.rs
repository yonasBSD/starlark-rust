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

//! Derive macros for `StarlarkSerialize` and `StarlarkDeserialize`.
//!
//! By default, each field is serialized/deserialized through the starlark context
//! (`StarlarkSerialize::starlark_serialize` / `StarlarkDeserialize::starlark_deserialize`).
//!
//! Fields annotated with `#[starlark_pagable(pagable)]` use the pagable bridge instead
//! (`PagableSerialize::pagable_serialize(ctx.pagable())` /
//!  `PagableDeserialize::pagable_deserialize(ctx.pagable())`).

use proc_macro2::Ident;
use proc_macro2::TokenStream;
use quote::quote;
use quote::quote_spanned;
use syn::Attribute;
use syn::DeriveInput;
use syn::Fields;
use syn::Generics;
use syn::Index;
use syn::LitStr;
use syn::Token;
use syn::WherePredicate;
use syn::parse::ParseStream;
use syn::parse_macro_input;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;

#[derive(Default)]
struct FieldAttrs {
    /// Route this field through `PagableSerialize`/`PagableDeserialize`
    /// (`ctx.pagable()`) instead of the starlark context.
    pagable: bool,
    /// Skip this field during ser/de.
    ///
    /// - Serialize: emit nothing for this field.
    /// - Deserialize: produce the value via `Default::default()` (requires
    ///   `T: Default`) *or* via the expression provided in `skip_expr` if set.
    ///
    /// Useful for runtime-only fields such as caches, profiling data, or
    /// fields holding non-serializable handles, where `Option::None` /
    /// `Vec::new()` / etc. is the right "empty" value to restore.
    skip: bool,
    /// Custom expression used to reconstruct the field on deserialize when
    /// combined with `skip`. If `Some("Ty::any()")`, the generated
    /// deserialize code uses that expression instead of `Default::default()`.
    skip_expr: Option<String>,
}

#[derive(Default)]
struct TypeAttrs {
    /// Extra `where`-clause predicates appended to the generated impl's
    /// where clause. Same shape as serde's `#[serde(bound = "...")]`:
    /// a comma-separated list of `syn::WherePredicate`s, so projection
    /// predicates like `V::String: StarlarkSerialize` are valid.
    bound: Vec<WherePredicate>,
    /// Overrides the type the generated impls target. When set, the impls
    /// are emitted as `impl Trait for <impl_for>` with empty `<>` and no
    /// where clause. Field bodies switch to unqualified trait dispatch
    /// (`Trait::method(&self.x, ...)`) so they don't depend on the original
    /// generic parameters being in scope.
    impl_for: Option<String>,
}

fn extract_type_attrs(attrs: &[Attribute]) -> syn::Result<TypeAttrs> {
    syn::custom_keyword!(bound);
    syn::custom_keyword!(impl_for);

    let mut opts = TypeAttrs::default();

    for attr in attrs {
        if !attr.path().is_ident("starlark_pagable") {
            continue;
        }

        attr.parse_args_with(|input: ParseStream| {
            while !input.is_empty() {
                if input.peek(bound) {
                    input.parse::<bound>()?;
                    input.parse::<Token![=]>()?;
                    let s: LitStr = input.parse()?;
                    if !opts.bound.is_empty() {
                        return Err(input.error("`bound` was set twice"));
                    }
                    let predicates = s
                        .parse_with(Punctuated::<WherePredicate, Token![,]>::parse_terminated)
                        .map_err(|err| {
                            syn::Error::new(
                                s.span(),
                                format!("failed to parse `bound = \"...\"` as a comma-separated list of `where` predicates: {err}"),
                            )
                        })?;
                    opts.bound = predicates.into_iter().collect();
                } else if input.peek(impl_for) {
                    input.parse::<impl_for>()?;
                    input.parse::<Token![=]>()?;
                    let s: LitStr = input.parse()?;
                    if opts.impl_for.is_some() {
                        return Err(input.error("`impl_for` was set twice"));
                    }
                    opts.impl_for = Some(s.value());
                } else {
                    return Err(input.error(
                        "expected `bound = \"...\"` or `impl_for = \"...\"` at the type level",
                    ));
                }
                if input.is_empty() {
                    break;
                }
                input.parse::<Token![,]>()?;
            }
            Ok(())
        })?;
    }

    Ok(opts)
}

/// Build the `<...>` token stream for the generated impl.
///
/// - `impl_for = "..."`: the type's original generic parameters are no
///   longer in scope (the impl targets a concrete type), so the impl
///   generics default to empty `<>`.
/// - Otherwise: the type's own `<...>` is propagated. (The user's `bound`
///   predicates land in the where clause, not here — see [`gen_target_ty`].)
fn gen_impl_generics(generics: &Generics, attrs: &TypeAttrs) -> syn::Result<TokenStream> {
    if attrs.impl_for.is_some() {
        return Ok(quote! { <> });
    }
    let (ig, _, _) = generics.split_for_impl();
    Ok(quote! { #ig })
}

/// Build the impl-target type and the trailing where-clause.
///
/// When `#[starlark_pagable(impl_for = "...")]` is present, the user-supplied
/// type is used verbatim and the original where-clause is dropped (it would
/// reference the original generic params that no longer exist on this impl).
///
/// Otherwise the impl is generic over the type's own parameters, and the
/// where clause is the type's own where clause with any `bound` predicates
/// appended.
fn gen_target_ty(
    name: &Ident,
    generics: &Generics,
    attrs: &TypeAttrs,
) -> syn::Result<(TokenStream, TokenStream)> {
    if let Some(t) = &attrs.impl_for {
        let toks: TokenStream = t.parse()?;
        return Ok((toks, quote! {}));
    }
    let (_, ty_generics, _) = generics.split_for_impl();
    let where_clause = build_where_clause(generics, &attrs.bound);
    Ok((quote! { #name #ty_generics }, where_clause))
}

/// Combine the type's own where-clause predicates with any user-supplied
/// `bound = "..."` predicates and emit a `where ...` token stream (or
/// nothing if both are empty).
fn build_where_clause(generics: &Generics, extra: &[WherePredicate]) -> TokenStream {
    let own = generics
        .where_clause
        .as_ref()
        .map(|w| w.predicates.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    if own.is_empty() && extra.is_empty() {
        return quote! {};
    }
    let predicates = own.iter().chain(extra.iter());
    quote! { where #(#predicates,)* }
}

fn extract_field_attrs(attrs: &[Attribute]) -> syn::Result<FieldAttrs> {
    syn::custom_keyword!(pagable);
    syn::custom_keyword!(skip);

    let mut opts = FieldAttrs::default();

    for attr in attrs {
        if !attr.path().is_ident("starlark_pagable") {
            continue;
        }

        attr.parse_args_with(|input: ParseStream| {
            if input.parse::<pagable>().is_ok() {
                if opts.pagable {
                    return Err(input.error("`pagable` was set twice"));
                }
                opts.pagable = true;
            } else if input.parse::<skip>().is_ok() {
                if opts.skip {
                    return Err(input.error("`skip` was set twice"));
                }
                opts.skip = true;
                // Optional `= "expr"` to customize the deserialize-side value
                // (falls back to `Default::default()` if omitted).
                if input.peek(Token![=]) {
                    input.parse::<Token![=]>()?;
                    let s: LitStr = input.parse()?;
                    opts.skip_expr = Some(s.value());
                }
            }
            if !input.is_empty() {
                return Err(input.error(format!(
                    "invalid attribute `{}`, expected `pagable`, `skip`, or `skip = \"expr\"` in `#[starlark_pagable(...)]`",
                    input
                )));
            }
            Ok(())
        })?;
    }

    Ok(opts)
}

fn field_ident(i: usize, field: &syn::Field) -> proc_macro2::TokenStream {
    match &field.ident {
        Some(name) => quote! { #name },
        None => {
            let idx = Index::from(i);
            quote! { #idx }
        }
    }
}

pub fn derive_starlark_pagable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ser = derive_starlark_serialize_impl(&input);
    let de = derive_starlark_deserialize_impl(&input);
    match (ser, de) {
        (Ok(ser), Ok(de)) => quote! { #ser #de }.into(),
        (Err(err), _) | (_, Err(err)) => err.to_compile_error().into(),
    }
}

pub fn derive_starlark_serialize(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_starlark_serialize_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn derive_starlark_serialize_impl(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let type_attrs = extract_type_attrs(&input.attrs)?;
    let impl_generics = gen_impl_generics(&input.generics, &type_attrs)?;
    let concrete = type_attrs.impl_for.is_some();
    let (target_ty, where_clause) = gen_target_ty(name, &input.generics, &type_attrs)?;

    let body = match &input.data {
        syn::Data::Struct(data) => gen_serialize_fields(&data.fields, concrete)?,
        syn::Data::Enum(data) => gen_serialize_enum(name, data, concrete)?,
        syn::Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                input,
                "StarlarkSerialize derive does not support unions",
            ));
        }
    };

    Ok(quote! {
        impl #impl_generics starlark::pagable::StarlarkSerialize for #target_ty #where_clause {
            fn starlark_serialize(
                &self,
                ctx: &mut dyn starlark::pagable::StarlarkSerializeContext,
            ) -> starlark::Result<()> {
                #body
                Ok(())
            }
        }
    })
}

fn gen_serialize_fields(fields: &Fields, concrete: bool) -> syn::Result<proc_macro2::TokenStream> {
    let mut stmts = Vec::new();

    for (i, field) in fields.iter().enumerate() {
        let attrs = extract_field_attrs(&field.attrs)?;
        let ident = field_ident(i, field);
        let ty = &field.ty;

        if attrs.skip {
            // Referenced to suppress "unused field" lints for skipped fields.
            stmts.push(quote_spanned! { field.span()=>
                let _ = &self.#ident;
            });
            continue;
        }

        let stmt = if attrs.pagable {
            quote_spanned! { field.span()=>
                pagable::PagableSerialize::pagable_serialize(&self.#ident, ctx.pagable())?;
            }
        } else if concrete {
            quote_spanned! { field.span()=>
                starlark::pagable::StarlarkSerialize::starlark_serialize(&self.#ident, ctx)?;
            }
        } else {
            quote_spanned! { field.span()=>
                <#ty as starlark::pagable::StarlarkSerialize>::starlark_serialize(&self.#ident, ctx)?;
            }
        };
        stmts.push(stmt);
    }

    Ok(quote! { #(#stmts)* })
}

pub fn derive_starlark_deserialize(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match derive_starlark_deserialize_impl(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn derive_starlark_deserialize_impl(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let type_attrs = extract_type_attrs(&input.attrs)?;
    let impl_generics = gen_impl_generics(&input.generics, &type_attrs)?;
    let concrete = type_attrs.impl_for.is_some();
    let (target_ty, where_clause) = gen_target_ty(name, &input.generics, &type_attrs)?;

    let body = match &input.data {
        syn::Data::Struct(data) => gen_deserialize_struct(name, &data.fields, concrete)?,
        syn::Data::Enum(data) => gen_deserialize_enum(name, data, concrete)?,
        syn::Data::Union(_) => {
            return Err(syn::Error::new_spanned(
                input,
                "StarlarkDeserialize derive does not support unions",
            ));
        }
    };

    Ok(quote! {
        impl #impl_generics starlark::pagable::StarlarkDeserialize for #target_ty #where_clause {
            fn starlark_deserialize(
                ctx: &mut dyn starlark::pagable::StarlarkDeserializeContext<'_>,
            ) -> starlark::Result<Self> {
                #body
            }
        }
    })
}

fn gen_skip_value(
    attrs: &FieldAttrs,
    ty: &syn::Type,
    span: proc_macro2::Span,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some(expr_str) = &attrs.skip_expr {
        let expr: TokenStream = expr_str.parse()?;
        Ok(quote_spanned! { span=> #expr })
    } else {
        Ok(quote_spanned! { span=>
            <#ty as ::std::default::Default>::default()
        })
    }
}

fn gen_deserialize_struct(
    name: &Ident,
    fields: &Fields,
    concrete: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    match fields {
        Fields::Named(named) => {
            let mut field_inits = Vec::new();
            for field in &named.named {
                let attrs = extract_field_attrs(&field.attrs)?;
                let ident = field.ident.as_ref().unwrap();
                let ty = &field.ty;

                let value = if attrs.skip {
                    gen_skip_value(&attrs, ty, field.span())?
                } else if attrs.pagable {
                    quote_spanned! { field.span()=>
                        <#ty as pagable::PagableDeserialize>::pagable_deserialize(ctx.pagable())?
                    }
                } else if concrete {
                    quote_spanned! { field.span()=>
                        starlark::pagable::StarlarkDeserialize::starlark_deserialize(ctx)?
                    }
                } else {
                    quote_spanned! { field.span()=>
                        <#ty as starlark::pagable::StarlarkDeserialize>::starlark_deserialize(ctx)?
                    }
                };
                field_inits.push(quote! { #ident: #value });
            }
            Ok(quote! {
                Ok(#name { #(#field_inits,)* })
            })
        }
        Fields::Unnamed(unnamed) => {
            let mut field_values = Vec::new();
            for field in &unnamed.unnamed {
                let attrs = extract_field_attrs(&field.attrs)?;
                let ty = &field.ty;

                let value = if attrs.skip {
                    gen_skip_value(&attrs, ty, field.span())?
                } else if attrs.pagable {
                    quote_spanned! { field.span()=>
                        <#ty as pagable::PagableDeserialize>::pagable_deserialize(ctx.pagable())?
                    }
                } else if concrete {
                    quote_spanned! { field.span()=>
                        starlark::pagable::StarlarkDeserialize::starlark_deserialize(ctx)?
                    }
                } else {
                    quote_spanned! { field.span()=>
                        <#ty as starlark::pagable::StarlarkDeserialize>::starlark_deserialize(ctx)?
                    }
                };
                field_values.push(value);
            }
            Ok(quote! {
                Ok(#name(#(#field_values,)*))
            })
        }
        Fields::Unit => Ok(quote! { Ok(#name) }),
    }
}

/// Generate `StarlarkSerialize` body for an enum.
///
/// Wire format: `u8` discriminant tag (variant index, 0..n) followed by the
/// variant's payload fields in declaration order. Variant field attributes
/// (`#[starlark_pagable(pagable|skip|skip = "...")]`) apply as with structs.
fn gen_serialize_enum(
    enum_name: &Ident,
    data: &syn::DataEnum,
    concrete: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    if data.variants.len() > 255 {
        return Err(syn::Error::new_spanned(
            enum_name,
            "StarlarkSerialize derive does not support enums with more than 255 variants",
        ));
    }

    let mut arms = Vec::new();
    for (idx, variant) in data.variants.iter().enumerate() {
        let tag = idx as u8;
        let variant_name = &variant.ident;

        match &variant.fields {
            Fields::Unit => {
                arms.push(quote! {
                    #enum_name::#variant_name => {
                        <u8 as pagable::PagableSerialize>::pagable_serialize(&#tag, ctx.pagable())?;
                    }
                });
            }
            Fields::Unnamed(unnamed) => {
                let bindings: Vec<_> = (0..unnamed.unnamed.len())
                    .map(|i| Ident::new(&format!("f{i}"), proc_macro2::Span::call_site()))
                    .collect();
                let mut stmts = Vec::new();
                for (binding, field) in bindings.iter().zip(unnamed.unnamed.iter()) {
                    let attrs = extract_field_attrs(&field.attrs)?;
                    let ty = &field.ty;
                    if attrs.skip {
                        stmts.push(quote_spanned! { field.span()=> let _ = #binding; });
                        continue;
                    }
                    if attrs.pagable {
                        stmts.push(quote_spanned! { field.span()=>
                            pagable::PagableSerialize::pagable_serialize(#binding, ctx.pagable())?;
                        });
                    } else if concrete {
                        stmts.push(quote_spanned! { field.span()=>
                            starlark::pagable::StarlarkSerialize::starlark_serialize(#binding, ctx)?;
                        });
                    } else {
                        stmts.push(quote_spanned! { field.span()=>
                            <#ty as starlark::pagable::StarlarkSerialize>::starlark_serialize(#binding, ctx)?;
                        });
                    }
                }
                arms.push(quote! {
                    #enum_name::#variant_name(#(#bindings),*) => {
                        <u8 as pagable::PagableSerialize>::pagable_serialize(&#tag, ctx.pagable())?;
                        #(#stmts)*
                    }
                });
            }
            Fields::Named(named) => {
                let bindings: Vec<_> = named
                    .named
                    .iter()
                    .map(|f| f.ident.as_ref().unwrap().clone())
                    .collect();
                let mut stmts = Vec::new();
                for (binding, field) in bindings.iter().zip(named.named.iter()) {
                    let attrs = extract_field_attrs(&field.attrs)?;
                    let ty = &field.ty;
                    if attrs.skip {
                        stmts.push(quote_spanned! { field.span()=> let _ = #binding; });
                        continue;
                    }
                    if attrs.pagable {
                        stmts.push(quote_spanned! { field.span()=>
                            pagable::PagableSerialize::pagable_serialize(#binding, ctx.pagable())?;
                        });
                    } else if concrete {
                        stmts.push(quote_spanned! { field.span()=>
                            starlark::pagable::StarlarkSerialize::starlark_serialize(#binding, ctx)?;
                        });
                    } else {
                        stmts.push(quote_spanned! { field.span()=>
                            <#ty as starlark::pagable::StarlarkSerialize>::starlark_serialize(#binding, ctx)?;
                        });
                    }
                }
                arms.push(quote! {
                    #enum_name::#variant_name { #(#bindings),* } => {
                        <u8 as pagable::PagableSerialize>::pagable_serialize(&#tag, ctx.pagable())?;
                        #(#stmts)*
                    }
                });
            }
        }
    }

    Ok(quote! {
        match self {
            #(#arms)*
        }
    })
}

/// Generate `StarlarkDeserialize` body for an enum.
fn gen_deserialize_enum(
    enum_name: &Ident,
    data: &syn::DataEnum,
    concrete: bool,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut arms = Vec::new();
    for (idx, variant) in data.variants.iter().enumerate() {
        let tag = idx as u8;
        let variant_name = &variant.ident;

        match &variant.fields {
            Fields::Unit => {
                arms.push(quote! {
                    #tag => Ok(#enum_name::#variant_name),
                });
            }
            Fields::Unnamed(unnamed) => {
                let mut values = Vec::new();
                for field in &unnamed.unnamed {
                    let attrs = extract_field_attrs(&field.attrs)?;
                    let ty = &field.ty;
                    let v = if attrs.skip {
                        gen_skip_value(&attrs, ty, field.span())?
                    } else if attrs.pagable {
                        quote_spanned! { field.span()=>
                            <#ty as pagable::PagableDeserialize>::pagable_deserialize(ctx.pagable())?
                        }
                    } else if concrete {
                        quote_spanned! { field.span()=>
                            starlark::pagable::StarlarkDeserialize::starlark_deserialize(ctx)?
                        }
                    } else {
                        quote_spanned! { field.span()=>
                            <#ty as starlark::pagable::StarlarkDeserialize>::starlark_deserialize(ctx)?
                        }
                    };
                    values.push(v);
                }
                arms.push(quote! {
                    #tag => Ok(#enum_name::#variant_name(#(#values),*)),
                });
            }
            Fields::Named(named) => {
                let mut inits = Vec::new();
                for field in &named.named {
                    let attrs = extract_field_attrs(&field.attrs)?;
                    let ident = field.ident.as_ref().unwrap();
                    let ty = &field.ty;
                    let v = if attrs.skip {
                        gen_skip_value(&attrs, ty, field.span())?
                    } else if attrs.pagable {
                        quote_spanned! { field.span()=>
                            <#ty as pagable::PagableDeserialize>::pagable_deserialize(ctx.pagable())?
                        }
                    } else if concrete {
                        quote_spanned! { field.span()=>
                            starlark::pagable::StarlarkDeserialize::starlark_deserialize(ctx)?
                        }
                    } else {
                        quote_spanned! { field.span()=>
                            <#ty as starlark::pagable::StarlarkDeserialize>::starlark_deserialize(ctx)?
                        }
                    };
                    inits.push(quote! { #ident: #v });
                }
                arms.push(quote! {
                    #tag => Ok(#enum_name::#variant_name { #(#inits),* }),
                });
            }
        }
    }

    let enum_name_str = enum_name.to_string();
    Ok(quote! {
        let tag = <u8 as pagable::PagableDeserialize>::pagable_deserialize(ctx.pagable())?;
        match tag {
            #(#arms)*
            _ => Err(starlark::Error::new_other(anyhow::anyhow!(
                "invalid {} variant tag: {}",
                #enum_name_str,
                tag,
            ))),
        }
    })
}
