# Smart Parking Space Detection

Hệ thống phát hiện và giám sát tình trạng ô đỗ xe từ video bằng **YOLO
(Ultralytics)** và **ByteTrack**, kết hợp bản đồ các ô đỗ được lưu trong
JSON. Chương trình tạo video kết quả có màu trạng thái ô đỗ và ghi log
để hỗ trợ phân tích hiện tượng trạng thái thay đổi không ổn định
(*flicker*).

> **Lưu ý:** Repository cần có mã nguồn Python tương ứng
> (`parking_monitor_styled.py`, `overlay_slots.py`,
> `analyze_flicker.py`, `requirements.txt`), file trọng số mô hình và dữ
> liệu video/bản đồ ô đỗ. Notebook `carParking.ipynb` hiện sử dụng đường
> dẫn Google Colab/Google Drive; hãy cập nhật đường dẫn nếu chạy ở môi
> trường khác.

## Tính năng

-   Phát hiện phương tiện trong video bằng mô hình YOLO.
-   Theo dõi đối tượng qua các khung hình bằng tracker ByteTrack (nếu
    cấu hình chạy sử dụng tracker này).
-   Xác định ô đỗ có xe hay còn trống dựa trên vùng tọa độ từng ô trong
    `slots.json` hoặc `slots3.json`.
-   Hiển thị trạng thái ô đỗ bằng lớp phủ màu trên video; hiển thị số ô
    trống/đã sử dụng theo nhóm.
-   Kiểm tra vị trí và kích thước bản đồ ô đỗ so với video đầu vào.
-   Xuất video kết quả và CSV log để kiểm tra, đánh giá hiện tượng
    *flicker*.

## Công nghệ

-   Python
-   OpenCV
-   NumPy
-   Ultralytics YOLO
-   ByteTrack / thư viện tracker được cấu hình trong dự án
-   Google Colab và Google Drive (môi trường được sử dụng trong
    notebook)

## Cấu trúc thư mục đề xuất

``` text
carParking/
├── carParking.ipynb
├── requirements.txt
├── parking_monitor_styled.py
├── overlay_slots.py
├── analyze_flicker.py
├── slots.json
├── slots3.json
├── models/
│   └── best11n.pt
├── data/
│   └── carParking3.mp4
└── out/
    ├── result_styled3.mp4
    ├── log3.csv
    └── overlay_slots.png
```

Tên file đầu vào/đầu ra có thể khác tùy cấu hình chạy. Không cần commit
video lớn hoặc trọng số mô hình lên GitHub nếu chúng quá nặng hoặc không
có quyền chia sẻ; có thể ghi hướng dẫn tải riêng trong repository.

## Cài đặt

### Cách 1: Google Colab

1.  Tải `carParking.ipynb` lên Google Colab.

2.  Mount Google Drive và đặt dự án theo cấu trúc phù hợp với notebook,
    ví dụ `MyDrive/carParking/`.

3.  Đưa video vào thư mục `data/`, trọng số YOLO vào `models/`, và các
    file JSON tọa độ ô đỗ vào thư mục dự án.

4.  Cài đặt thư viện:

    ``` bash
    pip install -r requirements.txt
    pip install ultralytics supervision
    pip install lap==0.5.13
    ```

5.  Chạy lần lượt các cell trong notebook. Nếu notebook báo không tìm
    thấy file, hãy kiểm tra lại các biến đường dẫn như `PROJECT_DIR`,
    `VIDEO_PATH`, `SLOTS_PATH` và `slots_path`.

### Cách 2: Môi trường Python cục bộ

1.  Cài Python tương thích với các phiên bản thư viện trong
    `requirements.txt`.

2.  Tạo môi trường ảo (khuyến nghị) và cài thư viện:

    ``` bash
    python -m venv .venv
    ```

    Windows:

    ``` bash
    .venv\Scripts\activate
    ```

    macOS/Linux:

    ``` bash
    source .venv/bin/activate
    ```

    Sau đó:

    ``` bash
    pip install -r requirements.txt
    pip install ultralytics supervision lap==0.5.13
    ```

3.  Chỉnh các đường dẫn video, trọng số và file tọa độ trong lệnh chạy
    hoặc trong notebook cho phù hợp với máy tính.

## Chuẩn bị bản đồ ô đỗ

File `slots.json` hoặc `slots3.json` chứa danh sách các ô đỗ. Mỗi ô có
`id` và `coordinates`, trong đó `coordinates` là các điểm tọa độ pixel
trên khung hình video.

Ví dụ cấu trúc:

``` json
[
  {
    "id": 1,
    "coordinates": [
      [100, 120],
      [180, 120],
      [180, 200],
      [100, 200]
    ]
  }
]
```

Tọa độ phải khớp với độ phân giải và góc nhìn của video dùng để chạy.
Nếu thay video hoặc thay đổi độ phân giải/góc camera, hãy kiểm tra và
tạo lại bản đồ ô đỗ.

## Kiểm tra vị trí các ô

Notebook có cell tạo/chỉnh sửa bản đồ ô đỗ và script `overlay_slots.py`
để vẽ các đường bao lên một số frame của video.

Ví dụ:

``` bash
python overlay_slots.py \
  --video data/carParking3.mp4 \
  --slots slots3.json \
  --frames 0 300 \
  --save out/overlay_slots.png
```

Tham số: - `--video`: video đầu vào. - `--slots`: file JSON chứa tọa độ
ô đỗ. - `--frames`: chỉ số frame cần kiểm tra. - `--save`: đường dẫn lưu
ảnh kiểm tra.

Nếu các đường bao lệch khỏi vị trí ô đỗ, hãy kiểm tra lại file JSON
trước khi chạy nhận diện.

## Chạy nhận diện

Notebook hiện có cấu hình mẫu cho video thứ ba với các tham số gần như
sau:

``` text
--video data/carParking3.mp4
--weights models/best11n.pt
--slots slots3.json
--classes 0 car
--imgsz 1184
--conf 0.2
--theta 0.25
--window 5
--dwell 8
--scale 1.0
--out out/result_styled3.mp4
--log out/log3.csv
--groups 6
```

Đây là các giá trị được sử dụng trong một cell của notebook, không phải
bộ tham số tối ưu cho mọi video. Cách gọi lệnh độc lập phụ thuộc vào
việc bạn đã lưu phần mã chạy trong notebook thành
`parking_monitor_styled.py` và parser của script có hỗ trợ các tham số
trên hay chưa.

Kết quả dự kiến: - Video đã vẽ trạng thái các ô đỗ. - Thông tin số ô
trống/đã sử dụng trên thanh đầu video. - CSV log để kiểm tra trạng thái
theo thời gian.

## Phân tích hiện tượng trạng thái nhấp nháy

Notebook gọi `analyze_flicker.py` trên file log, ví dụ:

``` bash
python analyze_flicker.py --log out/log3.csv --theta 0.35
```

Hãy bảo đảm CSV log đã được tạo và đường dẫn đúng. Nếu trạng thái ô đỗ
chuyển đổi liên tục giữa trống và có xe, hãy kiểm tra: - Bản đồ tọa độ ô
đỗ có khớp với video không. - Bounding box của xe có quá lớn hoặc dao
động không. - Giá trị `conf`, `theta`, `window`, `dwell` và cấu hình
tracker. - Các cột trạng thái trong CSV log để xác định vấn đề đến từ
phát hiện xe hay logic xác định ô đỗ.

Thay đổi tham số cần được kiểm thử trên nhiều đoạn video, vì tăng độ ổn
định quá mức có thể khiến hệ thống phản hồi chậm khi xe thực sự đi vào
hoặc rời ô.

## Dữ liệu và quyền sử dụng

-   Video, file trọng số (`.pt`) và dữ liệu gán nhãn không được bao gồm
    trong notebook đính kèm.
-   Chỉ chia sẻ dữ liệu khi bạn có quyền sử dụng và phân phối.
-   Với file lớn, cân nhắc dùng Git LFS hoặc hướng dẫn tải từ nguồn lưu
    trữ phù hợp.
-   Không đưa thông tin cá nhân hoặc dữ liệu nhạy cảm vào repository
    công khai.

## Hạn chế

-   Kết quả phụ thuộc vào chất lượng video, góc camera, ánh sáng, độ
    phân giải và độ chính xác của bản đồ ô đỗ.
-   Các tham số hiện tại là cấu hình thử nghiệm trong notebook, chưa
    chứng minh là tối ưu trên mọi bãi đỗ.
-   README mô tả quy trình và cấu hình thể hiện trong notebook; hãy cập
    nhật lại khi tên script, cấu trúc thư mục hoặc cách chạy thực tế
    thay đổi.

## Đóng góp

Nếu muốn cải thiện dự án, bạn có thể thử nghiệm thêm cách ổn định trạng
thái ô đỗ, xử lý xe dài chiếm nhiều ô, và đánh giá độ chính xác trên các
video khác nhau.

## License

Chưa xác định. Hãy thêm file `LICENSE` phù hợp trước khi công bố
repository nếu bạn muốn quy định rõ quyền sử dụng và phân phối mã nguồn.
