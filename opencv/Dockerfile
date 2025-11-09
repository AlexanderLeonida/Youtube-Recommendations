FROM jjanzic/docker-python3-opencv:latest

WORKDIR /app

# Fix old Debian Buster repos and install required dependencies
RUN sed -i 's|deb.debian.org|archive.debian.org|g' /etc/apt/sources.list && \
    sed -i '/security.debian.org/d' /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main" >> /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends \
        cmake \
        g++ \
        tesseract-ocr \
        libtesseract-dev \
        libleptonica-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p build && cd build && cmake .. && make -j$(nproc)

CMD ["./build/main"]
