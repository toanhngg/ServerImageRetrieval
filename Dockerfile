# Sử dụng hình ảnh Node.js v18
FROM node:18

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép package.json và package-lock.json (nếu có)
COPY package*.json ./

# Cài đặt các phụ thuộc
RUN npm install

# Sao chép mã nguồn vào hình ảnh
COPY . .

EXPOSE 3001
# Chạy ứng dụng
CMD ["npm", "start"]
