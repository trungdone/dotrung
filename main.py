# Bước 1: Tải và Làm Sạch Dữ Liệu
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Đặt đường dẫn thư mục chứa các tệp CSV
data_dir = r"C:\Users\ASUS\Downloads\dữ liệu BPS"
print(os.getcwd())

# Tải dữ liệu từ các file CSV
customer_df = pd.read_csv(os.path.join(data_dir, 'customer.csv'))
sale_df = pd.read_csv(os.path.join(data_dir, 'Sale.csv'))
product_detail_df = pd.read_csv(os.path.join(data_dir, 'Product Detail.csv'))
product_group_df = pd.read_csv(os.path.join(data_dir, 'Product Group.csv'))
market_trend_df = pd.read_csv(os.path.join(data_dir, 'Market Trend.csv'))
website_access_category_df = pd.read_csv(os.path.join(data_dir, 'Website Access Category.csv'))

# Chuyển đổi cột ngày tháng sang định dạng datetime
sale_df['SaleDate'] = pd.to_datetime(sale_df['SaleDate'])

# Thêm cột SaleMonth vào dataframe sale_df
sale_df['SaleMonth'] = sale_df['SaleDate'].dt.to_period('M')  # Đoạn này được sửa để tạo cột 'SaleMonth'

# Hiển thị vài dòng đầu tiên của mỗi DataFrame
print(customer_df.head())
print(sale_df.head())
print(product_detail_df.head())
print(product_group_df.head())
print(market_trend_df.head())
print(website_access_category_df.head())

# Bước 2: Làm Sạch và Tiền Xử Lý Dữ Liệu
# Kiểm tra giá trị thiếu
print(customer_df.isnull().sum())
print(sale_df.isnull().sum())
print(product_detail_df.isnull().sum())
print(product_group_df.isnull().sum())
print(market_trend_df.isnull().sum())
print(website_access_category_df.isnull().sum())

# Chuyển các cột chứa giá trị thiếu về kiểu dữ liệu phù hợp trước khi thay thế
customer_df = customer_df.apply(lambda x: x.astype(str) if x.isnull().any() else x)
product_group_df = product_group_df.apply(lambda x: x.astype(str) if x.isnull().any() else x)

# Xử lý giá trị thiếu
customer_df.fillna('', inplace=True)
sale_df.dropna(subset=['CustomerID', 'ProductID', 'SaleDate'], inplace=True)
product_detail_df.fillna('', inplace=True)
product_group_df.fillna('', inplace=True)
market_trend_df.fillna('', inplace=True)
website_access_category_df.fillna('', inplace=True)

# Chuyển đổi kiểu dữ liệu nếu cần thiết
sale_df['SaleDate'] = pd.to_datetime(sale_df['SaleDate'], errors='coerce')
market_trend_df['TrendStartDate'] = pd.to_datetime(market_trend_df['TrendStartDate'], errors='coerce')
market_trend_df['TrendEndDate'] = pd.to_datetime(market_trend_df['TrendEndDate'], errors='coerce')
website_access_category_df['AccessDate'] = pd.to_datetime(website_access_category_df['AccessDate'], errors='coerce')

# Loại bỏ các hàng không thể chuyển đổi kiểu dữ liệu
sale_df.dropna(subset=['SaleDate'], inplace=True)
market_trend_df.dropna(subset=['TrendStartDate', 'TrendEndDate'], inplace=True)
website_access_category_df.dropna(subset=['AccessDate'], inplace=True)

# Hiển thị thông tin sau khi làm sạch dữ liệu
print(customer_df.info())
print(sale_df.info())
print(product_detail_df.info())
print(product_group_df.info())
print(market_trend_df.info())
print(website_access_category_df.info())

# Loại bỏ các hàng có giá trị âm trong cột Quantity của Sale Table
sale_df = sale_df[sale_df['Quantity'] >= 0]

# Chuẩn hóa dữ liệu
customer_df = customer_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
product_detail_df = product_detail_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
product_group_df = product_group_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
market_trend_df = market_trend_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
website_access_category_df = website_access_category_df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Lưu Dữ Liệu Đã Làm Sạch vào Các Tệp CSV Mới
customer_df.to_csv('customer_cleaned.csv', index=False)
sale_df.to_csv('sale_cleaned.csv', index=False)
product_detail_df.to_csv('product_detail_cleaned.csv', index=False)
product_group_df.to_csv('product_group_cleaned.csv', index=False)
market_trend_df.to_csv('market_trend_cleaned.csv', index=False)
website_access_category_df.to_csv('website_access_category_cleaned.csv', index=False)

# Bước 3: Trực Quan Hóa Dữ Liệu
# Doanh số theo thời gian
sale_df['SaleDate'].value_counts().sort_index().plot(kind='line')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Sales')
plt.show()

# Doanh số theo sản phẩm
product_sales = sale_df['ProductID'].value_counts().head(10)
product_sales.plot(kind='bar')
plt.title('Top 10 Best Selling Products')
plt.xlabel('Product ID')
plt.ylabel('Number of Sales')
plt.show()

# Doanh số theo khách hàng
customer_sales = sale_df['CustomerID'].value_counts().head(10)
customer_sales.plot(kind='bar')
plt.title('Top 10 Customers by Sales Volume')
plt.xlabel('Customer ID')
plt.ylabel('Number of Sales')
plt.show()

# Xu hướng thị trường theo yếu tố tác động
market_trend_df['ImpactFactor'].value_counts().plot(kind='bar')
plt.title('Market Trends by Impact Factor')
plt.xlabel('Impact Factor')
plt.ylabel('Count')
plt.show()

# Total Sales Over Time
sale_df.set_index('SaleDate')['TotalPrice'].resample('ME').sum().plot(kind='line')
plt.title('Total Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# Monthly Sales Trends
# Đoạn này được sửa lại để lấy đúng cột 'SaleMonth'
sale_df.groupby(sale_df['SaleDate'].dt.to_period('M'))['TotalPrice'].sum().plot(kind='line')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

# Bước 4: Xây Dựng Mô Hình Máy Học
# Chuẩn Bị Dữ Liệu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Xây dựng đặc trưng
sale_df['SaleYear'] = sale_df['SaleDate'].dt.year
sale_df['SaleMonth'] = sale_df['SaleDate'].dt.month
features = sale_df[['SaleYear', 'SaleMonth']]
target = sale_df['TotalPrice']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán
predictions = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Trực quan hóa kết quả dự đoán
plt.scatter(y_test, predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Comparison of Actual and Predicted Values')
plt.show()

# Lưu dữ liệu dự đoán vào DataFrame
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# Hiển thị 10 dòng đầu tiên của DataFrame dự đoán
print(predictions_df.head())

import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame for the actual and predicted values
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# Trực quan hóa dữ liệu dự đoán và thực tế
plt.figure(figsize=(10, 5))
plt.plot(predictions_df['Actual'].values, label='Actual Values', marker='o')
plt.plot(predictions_df['Predicted'].values, label='Predicted Values', marker='x')
plt.title('Future Sales Predictions')
plt.xlabel('Sample')
plt.ylabel('Sales')
plt.legend()
plt.show()











