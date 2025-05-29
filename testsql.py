import psycopg2

# Cấu hình thông tin kết nối đến PostgreSQL
DB_CONFIG = {
    'user': 'postgres',
    'host': 'localhost',
    'database': 'doantotnghiep',
    'password': '123456',  # Thay đổi nếu mật khẩu khác
    'port': 5432
}

try:
    # Thử kết nối
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ Kết nối thành công đến cơ sở dữ liệu PostgreSQL!")
    conn.close()
except Exception as e:
    print("❌ Kết nối thất bại. Lỗi:")
    print(e)
