<div align="center">
  <h1>🧠 SmartRec: Hệ thống Gợi ý Sản phẩm Cá nhân hóa</h1>
  <h3>Dự án Ứng dụng Deep Learning trong E-Commerce Recommendation Systems</h3>
</div>

---

## 📖 Giới thiệu (Abstract)
**SmartRec** là một hệ thống Trí tuệ Nhân tạo (Machine Learning) được thiết kế để giải quyết bài toán gợi ý cá nhân hóa dựa trên học sâu.

Hệ thống giải quyết bài toán cốt lõi của Thương mại điện tử: **Gợi ý cá nhân hóa trong điều kiện thiếu vắng "Đánh giá trực tiếp" (Explicit Rating)**. Thay vì phụ thuộc vào việc người dùng chấm 1-5 sao, SmartRec khai phá luồng dữ liệu **Clickstream Hành vi (Implicit Feedback)** bao gồm các luồng sự kiện `View`, `Add_to_cart`, và `Purchase`.

Dự án này là một chuỗi đường ống (Data Pipeline) hoàn chỉnh, từ bước Thu thập Dữ liệu (ETL) ➔ Huấn luyện Mô hình Học sâu (Model Training) ➔ Triển khai Hệ thống Suy luận (Inference API) ➔ Trực quan hóa kết quả (React Dashboard).

---

## 📸 Giao diện Ứng dụng (System Screenshots)

Dưới đây là một số hình ảnh thực tế của hệ thống:

<details>
<summary><b>1. System Overview (Dashboard MLOps)</b></summary>
<br>
<em>Trực quan hóa độ trễ suy luận, lưu lượng truy cập thực tế và trạng thái mô hình học sâu đang được triển khai (NCF-v2.1).</em>
<br><br>
<img src="docs/images/dashboard.png" alt="Dashboard MLOps" width="800"/>
</details>

<details>
<summary><b>2. Khám phá Sản phẩm (Product Catalog)</b></summary>
<br>
<em>Giao diện khám phá kho sản phẩm được phân loại theo danh mục.</em>
<br><br>
<img src="docs/images/products.png" alt="Product Catalog" width="800"/>
</details>

<details>
<summary><b>3. Công cụ Suy luận cá nhân hóa & A/B Testing (Inference Engine)</b></summary>
<br>
<em>Trực quan hóa mức độ ưu tiên (Latent Affinity), giải thích logic bằng AI (XAI) và tính năng chuyển đổi A/B Model thời gian thực giữa MF và NCF.</em>
<br><br>
<img src="docs/images/recommendations.png" alt="Inference Engine" width="800"/>
</details>

<details>
<summary><b>4. Đánh giá Hiệu suất (Performance & Impact Metrics)</b></summary>
<br>
<em>Trình bày các chỉ số kinh doanh ước tính (Business Uplift), biểu đồ độ hội tụ mất mát (BCE Loss Convergence) và kết quả Ranking Top-K.</em>
<br><br>
<img src="docs/images/results.png" alt="Performance Metrics" width="800"/>
</details>

---

## 🏗 Kiến trúc MLOps & Luồng Dữ Liệu
Dự án được cấu trúc theo chuẩn Microservices/MLOps:

1. **Data Ingestion (Xử lý Dữ liệu):** Sử dụng tập dữ liệu công khai `MovieLens 100k`. Thông qua module `movielens_loader.py`, hệ thống tự động tải và chuyển đổi thuật toán 100,000 lượt rating (1-5 sao) thành các luồng hành vi E-commerce (`view`, `add_to_cart`, `purchase`) để giải quyết bài toán Data Sparsity (Dữ liệu thưa). Mẫu dữ liệu sau đó được chia theo **Temporal Split** (80% Train, 10% Val, 10% Test).
2. **Model Training (Huấn luyện Pytorch):** Dữ liệu được đẩy vào các DataLoader của PyTorch để cập nhật trọng số (weights) sinh ra các ma trận Embeddings không gian N-chiều.
3. **Inference Engine (Máy suy luận FastAPI):** Cung cấp API trực tiếp nạp (load) các file weight `.pth` vào RAM, thực hiện tính toán ma trận với độ trễ tối ưu (< 150ms/request) và bổ sung Explainable AI (XAI) để giải thích lý do gợi ý.
4. **Monitoring Dashboard:** Một giao diện ReactJS thời gian thực hiển thị dữ liệu giám sát về đo lường độ trễ (Latency), Affinity Scores, và hiệu suất của Model.

---

## 🔬 Kiến trúc Thuật toán (Core ML Architectures)

SmartRec triển khai và so sánh hai trường phái thuật toán chính trong Hệ thống gợi ý:

### 1. Matrix Factorization (MF-ALS / Baseline)
Đây là thuật toán Linear học máy kinh điển. Sử dụng một phép trích xuất đặc trưng (Latent Feature Extraction). 
Hệ thống ánh xạ cả `Users` và `Items` vào chung một không gian ẩn (Latent Space). Điểm số gợi ý $y_{ui}$ được tính toán dựa trên **tích vô hướng (dot-product)** của hai vector nhúng (Embeddings) này, cộng thêm các trọng số sai lệch (bias).

### 2. Neural Collaborative Filtering (NCF)
NCF là trọng tâm Deep Learning của đồ án này (Tham chiếu: *Xiangnan He et al., WWW 2017*).
Khắc phục điểm yếu phi tuyến tính của MF, kiến trúc NCF:
- Concatenate (Nối) vector User Embedding và Item Embedding lại với nhau.
- Đưa qua mạng **Multi-Layer Perceptron (MLP)** gồm các tầng ẩn (Hidden Layers): `128 -> 64 -> 32`.
- Sử dụng hàm kích hoạt ReLU để học các biểu diễn phi tuyến tính phức tạp.
- Tầng phân loại cuối sử dụng Sigmoid/BCE Loss để dự đoán tỉ lệ tương tác của người dùng.

### ❄️ Cơ chế Giải quyết Khởi động lạnh (Cold-Start & XAI)
Để xây dựng một hệ thống Recommendation hoàn chỉnh thực tế, giải quyết bài toán **Cold-Start** (Người dùng mới chưa từng mua hàng) là bắt buộc. 
SmartRec cài đặt cơ chế **Hybrid Content-Based Filtering**:
- Nếu hệ thống phát hiện User không có trong không gian Vector ID: Hệ thống hủy bỏ luồng Deep Learning, chuyển qua luồng quy tắc phân tích sở thích đồ thị.
- Tính toán độ phổ biến cục bộ (Local Popularity) trong chính tệp danh mục (Category) mà người dùng yêu thích, thay vì lấy Popularity toàn cục (Global Popularity).

---

## 📊 Kết quả Đánh giá Mô hình (Evaluation Metrics)
Hệ thống sử dụng các thang đo Tiêu chuẩn hóa trong lĩnh vực Information Retrieval cho Bài toán Ranking Top-K (K=10):

| Thuật toán | Validation Loss | RMSE (Sai số suy luận) | Latency (Độ trễ trung bình) |
|------------|-----------------|-------------------------|-----------------------------|
| Matrix Factorization | 1.8415 | 2.1220 | ~80ms |
| **Neural Collaborative Filtering** | **1.3217** | **1.2117** | **~142ms** |

*(Ghi chú: NCF chứng minh năng lực học sâu biểu diễn tốt hơn đáng kể so với MF thông thường bằng sự sụt giảm mạnh trong Validation Loss).*

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### Yêu cầu Hệ thống
- **Backend:** Python 3.10+, PyTorch 2.x, FastAPI. (Khuyến nghị có CUDA/GPU để Train model nhanh hơn).
- **Frontend:** Node.js 18+, Vite, TailwindCSS.

### Bước 1: Chuẩn bị Backend & Pipeline Dữ liệu
```bash
cd backend
python -m venv venv
# Activate venv (Windows: venv\Scripts\activate | Lin/Mac: source venv/bin/activate)
pip install -r requirements.txt

# Chạy ETL Pipeline tải và xử lý Dữ liệu MovieLens tự động
cd data
python movielens_loader.py
```

### Bước 2: Huấn luyện Mô hình Machine Learning (Training)
```bash
cd backend
python training/train.py
```
*Script sẽ tự động train 2 thuật toán MF và NCF qua 50 Epochs (hoặc Early Stopping), lưu file Weights `.pth` vào thư mục `checkpoints/`.*

### Bước 3: Khởi chạy Máy chủ API (Inference Engine)
```bash
cd backend
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Truy cập Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Bước 4: Khởi chạy AI Dashboard (Frontend)
(Mở một Terminal mới)
```bash
npm install
npm run dev
```
- Mở Dashboard phân tích tại: [http://localhost:5173](http://localhost:5173)

---

## 📜 Cấu trúc Code cốt lõi

- Thiết kế Kiến trúc mạng Deep Learning (NCF/MLP): `backend/models/ncf.py`
- Bộ tải & Tiền xử lý Dữ liệu thô (ETL Pipeline): `backend/data/movielens_loader.py`
- Tiêm mô hình vào RAM và Engine Suggester: `backend/api/inference.py`
- Loss function & Backpropagation Training Loop: `backend/training/train.py`
- Trực quan hóa Điểm Vector và XAI (Explainable AI): `src/pages/Recommendations.tsx`

---

> Dự án mã nguồn mở - Bản quyền thuộc về tác giả.