# sys1-protocol

[简体中文](README.zh-CN.md)

Official System One wire types and validation for TypeSafe-compatible services.

This crate is the single source of truth for `POST /v1/systemone` and
`GET /v1/models`. It has no HTTP or inference dependencies and is shared by the
SDK, server, hub, and tests.

## What is included

- `SystemOneRequest` / `SystemOneResponse`
- `State`, `Question`, `Answer`, `Content`
- `ChoiceQuestion`, `ScoreQuestion`, `NoulQuestion`, `BoxQuestion`
- `ChoiceAnswer`, `ScoreAnswer`, `NoulAnswer`, `BoxAnswer`
- `ValidationError`, `ValidationReport`, `ErrorResponse`
- `ModelMetadataList` for `/v1/models`

## Example

```rust
use sys1_protocol::{parse_request_slice, SystemOneResponse};

let request = parse_request_slice(body_bytes)?;
let response = SystemOneResponse {
    model: "laya-multilingual".to_owned(),
    answers, // IndexMap<String, Answer>
    usage,
};
```

`parse_request_slice` returns `ValidationReport` on failure, ready to serialize
as `422 {"detail": [...]}`. Unknown question types, invalid state types, empty
questions, invalid criteria counts and malformed assets are rejected.

## Wire rules

- `state`: string, object, or array; null, number, and boolean are rejected.
- `questions`: at least one; `choice` criteria 1..=255; `score` criteria 1..=10.
- `instructions` and choice/noul criteria values accept string, object, array, or null.
- Answers use the official `type` discriminator and field names.
- `/v1/models` requires `name`, `description`, and `release_date`.

## Test

```bash
cargo test --offline -p sys1-protocol
```

The tests cover the official noul/choice/score/box examples, error-body shapes,
limit validation, and round-tripping.
