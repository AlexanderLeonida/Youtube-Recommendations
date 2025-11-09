# Use Node 20
FROM node:20

WORKDIR /app

# Copy package files first for caching
COPY package*.json ./

# Install dependencies with ajv fixes
RUN npm install ajv@6.12.6 ajv-keywords@3.5.2 --legacy-peer-deps

# Now install the rest
RUN npm install --legacy-peer-deps

# Copy the rest of the app
COPY . .

EXPOSE 3000
CMD ["npm", "start"]
