# Multi-stage build for lauturgie-server, the scv1-wire-compatible HTTP API.
# cargo-chef caches the dependency build (axum/tokio/etc.) in its own layer so
# editing src/ doesn't recompile the world.

FROM rust:1.96-slim-trixie AS chef
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# --features server so the optional axum/tokio/serde deps are cooked too.
RUN cargo chef cook --release --features server --recipe-path recipe.json
COPY . .
RUN cargo build --release --features server --bin lauturgie-server

# Runtime: trixie-slim matches the builder's glibc ABI. The binary has no TLS
# or other system-lib deps, so plain libc is all it needs.
FROM debian:trixie-slim AS runtime
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin lauturgie
COPY --from=builder /app/target/release/lauturgie-server /usr/local/bin/lauturgie-server
USER lauturgie
ENV PORT=8080
EXPOSE 8080
CMD ["lauturgie-server"]
