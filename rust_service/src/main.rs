use actix_web::{web, App, HttpServer, HttpResponse, Responder};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct PricePredictionInput {
    lbmp: f64,
    marginal_cost_losses: f64,
    marginal_cost_congestion: f64,
}

async fn predict(data: web::Json<Vec<PricePredictionInput>>) -> impl Responder {
    let client = Client::new();
    let res = client.post("http://127.0.0.1:8000/predict/")
        .json(&data.0)
        .send()
        .await;

    match res {
        Ok(response) => {
            if let Ok(json) = response.json::<serde_json::Value>().await {
                HttpResponse::Ok().json(json)
            } else {
                HttpResponse::InternalServerError().body("Failed to parse response")
            }
        }
        Err(_) => HttpResponse::InternalServerError().body("Failed to call prediction service"),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/predict", web::post().to(predict))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
