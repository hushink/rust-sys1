# sys1-protocol

[English](README.md)

TypeSafe-compatible 服务的官方 System One wire types 与校验。

这个 crate 是 `POST /v1/systemone` 与 `GET /v1/models` 的唯一契约来源，不依赖
HTTP 或推理栈，被 SDK、server、hub 与测试共用。

## 包含内容

- `SystemOneRequest` / `SystemOneResponse`
- `State`、`Question`、`Answer`、`Content`
- `ChoiceQuestion`、`ScoreQuestion`、`NoulQuestion`、`BoxQuestion`
- `ChoiceAnswer`、`ScoreAnswer`、`NoulAnswer`、`BoxAnswer`
- `ValidationError`、`ValidationReport`、`ErrorResponse`
- `/v1/models` 的 `ModelMetadataList`

## 作为库使用

```toml
[dependencies]
sys1-protocol = { git = "https://github.com/hushink/rust-sys1" }
# 或从本地 checkout 引入：
# sys1-protocol = { path = "../rust-sys1/crates/sys1-protocol" }
```

只需要请求/响应解析与校验、不引入推理依赖时，可以单独使用本 crate。

## 示例

```rust
use sys1_protocol::{parse_request_slice, SystemOneResponse};

let request = parse_request_slice(body_bytes)?;
let response = SystemOneResponse {
    model: "laya-multilingual".to_owned(),
    answers, // IndexMap<String, Answer>
    usage,
};
```

`parse_request_slice` 失败时返回 `ValidationReport`，可直接序列化为
`422 {"detail": [...]}`。未知问题类型、非法 state 类型、空 questions、criteria
数量越界与非法 assets 都会被拒绝。

## Wire 规则

- `state`：string / object / array；拒绝 null、数字、布尔。
- `questions`：至少 1 个；`choice` criteria 1..=255；`score` criteria 1..=10。
- `instructions` 与 choice/noul criteria 值接受 string / object / array / null。
- answer 使用官方 `type` 判别字段与字段名。
- `/v1/models` 的 `name`、`description`、`release_date` 均为必填。

## 测试

```bash
cargo test --offline -p sys1-protocol
```

测试覆盖官方 noul/choice/score/box 示例、错误体形状、限额校验与往返序列化。
