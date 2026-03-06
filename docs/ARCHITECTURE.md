# SmartRec – Architecture & Development Notes

Tài liệu tham khảo kiến trúc và quy trình phát triển (không bắt buộc để chạy dự án).

## Kiến trúc tổng quan

```
React (Vite)  ←→  /api proxy  ←→  FastAPI (uvicorn:8000)  ←→  PyTorch (MF / NCF)
                         ↑
                  vite.config.ts hoặc server.ts
```

## Luồng dữ liệu

1. **Training:** `interactions.csv` + `users.json`, `products.json` → `dataset_loader` → train/val/test → `train.py` → checkpoints + `evaluation_results.json`.
2. **Inference:** Request `GET /api/recommendations/{user_id}` → `RecommendationEngine` load checkpoint → `model.recommend()` → JSON recommendations.

## Scripts bổ sung

- **Health check:** Từ thư mục gốc: `python health_check.py` – kiểm tra data, checkpoints, dependencies.
- **Verify data:** Trong `backend`: `python verify_data_matching.py` – tỷ lệ interaction khớp preference (data quality).
- **Generate data:** Trong `backend/data`: `python dataset_generator.py` – tạo lại users, products, interactions.

## Ports

- Backend: **8000**
- Vite dev: **5173** (proxy /api → 8000)
- Node server (dev:server): **3001** (proxy /api → 8000)
