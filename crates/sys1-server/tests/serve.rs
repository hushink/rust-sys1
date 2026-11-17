use std::sync::Arc;
use std::time::Duration;
use sys1_core::{
    DecisionModel, ModelError, ModelFactory, ModelManifest, ModelRegistry, RegistryConfig,
};
use sys1_server::{ServerConfig, ServerExtensions, serve_listener_with_extensions_and_shutdown};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct UnusedFactory;

impl ModelFactory for UnusedFactory {
    fn load(&self, _manifest: &ModelManifest) -> Result<Arc<dyn DecisionModel>, ModelError> {
        Err(ModelError::load("no models in this test"))
    }
}

#[tokio::test]
async fn graceful_shutdown_serves_then_stops_the_listener() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test server");
    let address = listener.local_addr().expect("local address");
    let registry = Arc::new(
        ModelRegistry::new(
            Vec::new(),
            Arc::new(UnusedFactory),
            RegistryConfig::default(),
        )
        .expect("registry"),
    );
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(serve_listener_with_extensions_and_shutdown(
        listener,
        registry,
        ServerConfig {
            batching: None,
            ..ServerConfig::default()
        },
        ServerExtensions::default(),
        async move {
            let _ = shutdown_rx.await;
        },
    ));

    let mut stream = tokio::net::TcpStream::connect(address)
        .await
        .expect("connect");
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .expect("write request");
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("read response");
    let response = String::from_utf8_lossy(&response);
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");

    shutdown_tx.send(()).expect("request shutdown");
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("server did not shut down")
        .expect("server task")
        .expect("serve");
    assert!(tokio::net::TcpStream::connect(address).await.is_err());
}
