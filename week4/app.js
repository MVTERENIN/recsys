class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.userIdMap = new Map();
        this.itemIdMap = new Map();
        this.reverseUserIdMap = new Map();
        this.reverseItemIdMap = new Map();
        this.userRatedItems = new Map();
        this.model = null;
        
        this.initializeEventListeners();
        this.setupLossChart();
        this.setupEmbeddingChart();
    }

    initializeEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('trainModel').addEventListener('click', () => this.trainModel());
        document.getElementById('testModel').addEventListener('click', () => this.testModel());
    }

    async loadData() {
        this.updateStatus('Loading MovieLens 100K data...');
        
        try {
            await this.loadUsers();
            await this.loadItems();
            await this.loadInteractions();
            this.preprocessData();
            this.updateStatus(`Data loaded successfully: ${this.users.size} users, ${this.items.size} movies, ${this.interactions.length} ratings`);
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    async loadUsers() {
        const response = await fetch('data/u.user');
        const text = await response.text();
        const lines = text.trim().split('\n');
        
        lines.forEach(line => {
            const [userId, age, gender, occupation, zipCode] = line.split('|');
            this.users.set(parseInt(userId), {
                age: parseInt(age),
                gender,
                occupation,
                zipCode
            });
        });
    }

    async loadItems() {
        const response = await fetch('data/u.item');
        const text = await response.text();
        const lines = text.trim().split('\n');
        
        lines.forEach(line => {
            const parts = line.split('|');
            const itemId = parseInt(parts[0]);
            const title = parts[1];
            const releaseDate = parts[2];
            const genres = parts.slice(5, 24).map(g => parseInt(g));
            
            // Extract year from title (format: "Movie Name (YYYY)")
            const yearMatch = title.match(/\((\d{4})\)$/);
            const year = yearMatch ? parseInt(yearMatch[1]) : null;
            
            this.items.set(itemId, {
                title,
                year,
                genres,
                releaseDate
            });
        });
    }

    async loadInteractions() {
        const response = await fetch('data/u.data');
        const text = await response.text();
        const lines = text.trim().split('\n');
        
        lines.forEach(line => {
            const [userId, itemId, rating, timestamp] = line.split('\t');
            this.interactions.push({
                userId: parseInt(userId),
                itemId: parseInt(itemId),
                rating: parseFloat(rating),
                timestamp: parseInt(timestamp)
            });
        });
    }

    preprocessData() {
        // Create user and item indices
        let userIdx = 0;
        let itemIdx = 0;
        
        this.users.forEach((user, userId) => {
            this.userIdMap.set(userId, userIdx);
            this.reverseUserIdMap.set(userIdx, userId);
            userIdx++;
        });
        
        this.items.forEach((item, itemId) => {
            this.itemIdMap.set(itemId, itemIdx);
            this.reverseItemIdMap.set(itemIdx, itemId);
            itemIdx++;
        });

        // Build user->rated items map
        this.userRatedItems.clear();
        this.interactions.forEach(interaction => {
            const userId = interaction.userId;
            if (!this.userRatedItems.has(userId)) {
                this.userRatedItems.set(userId, []);
            }
            this.userRatedItems.get(userId).push({
                itemId: interaction.itemId,
                rating: interaction.rating,
                timestamp: interaction.timestamp
            });
        });

        // Sort each user's ratings by rating (desc) then timestamp (desc)
        this.userRatedItems.forEach((ratings, userId) => {
            ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        });
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }

    setupLossChart() {
        const canvas = document.getElementById('lossChart');
        this.lossCtx = canvas.getContext('2d');
        this.lossCtx.fillStyle = 'white';
        this.lossCtx.fillRect(0, 0, canvas.width, canvas.height);
        
        this.lossData = [];
        this.lossCtx.strokeStyle = 'blue';
        this.lossCtx.lineWidth = 2;
    }

    setupEmbeddingChart() {
        const canvas = document.getElementById('embeddingChart');
        this.embeddingCtx = canvas.getContext('2d');
        this.embeddingCtx.fillStyle = 'white';
        this.embeddingCtx.fillRect(0, 0, canvas.width, canvas.height);
    }

    updateLossChart(loss) {
        this.lossData.push(loss);
        
        const canvas = document.getElementById('lossChart');
        const ctx = this.lossCtx;
        const width = canvas.width;
        const height = canvas.height;
        
        // Clear and redraw
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);
        
        if (this.lossData.length < 2) return;
        
        const maxLoss = Math.max(...this.lossData);
        const minLoss = Math.min(...this.lossData);
        const range = maxLoss - minLoss || 1;
        
        ctx.beginPath();
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        
        this.lossData.forEach((loss, index) => {
            const x = (index / (this.lossData.length - 1)) * width;
            const y = height - ((loss - minLoss) / range) * height * 0.9 - 20;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Add labels
        ctx.fillStyle = 'black';
        ctx.font = '12px Arial';
        ctx.fillText(`Latest Loss: ${loss.toFixed(4)}`, 10, 20);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 40);
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, 60);
    }

    async trainModel() {
        if (this.interactions.length === 0) {
            this.updateStatus('Please load data first');
            return;
        }

        this.updateStatus('Initializing Two-Tower Model...');
        
        const numUsers = this.users.size;
        const numItems = this.items.size;
        const numGenres = 19; // MovieLens has 19 genres
        const numOccupations = new Set(Array.from(this.users.values()).map(u => u.occupation)).size;
        const numGenders = 2; // Assuming 'M' and 'F'

        this.model = new TwoTowerModel(
            numUsers,
            numItems,
            numGenres,
            numOccupations,
            numGenders,
            {
                embeddingDim: 32,
                hiddenUnits: [64, 32],
                learningRate: 0.001,
                batchSize: 256,
                maxInteractions: 80000
            }
        );

        this.updateStatus('Starting training...');
        this.lossData = [];
        
        const epochs = 20;
        for (let epoch = 0; epoch < epochs; epoch++) {
            this.updateStatus(`Training epoch ${epoch + 1}/${epochs}...`);
            const loss = await this.model.trainEpoch(this.interactions, this.userIdMap, this.itemIdMap, this.users, this.items);
            this.updateLossChart(loss);
            
            if ((epoch + 1) % 5 === 0) {
                this.updateStatus(`Epoch ${epoch + 1} completed. Loss: ${loss.toFixed(4)}`);
            }
        }
        
        this.updateStatus('Training completed! Generating embedding visualization...');
        await this.visualizeEmbeddings();
        this.updateStatus('Ready for testing! Click "Test Recommendations" to see results.');
    }

    async visualizeEmbeddings() {
        // Sample 200 items for visualization
        const sampleSize = Math.min(200, this.items.size);
        const sampledItemIndices = Array.from({length: sampleSize}, (_, i) => 
            Math.floor(i * (this.items.size / sampleSize))
        );
        
        const itemEmbeddings = [];
        const itemTitles = [];
        
        for (const itemIdx of sampledItemIndices) {
            const itemId = this.reverseItemIdMap.get(itemIdx);
            const item = this.items.get(itemId);
            const embedding = await this.model.getItemEmbedding(itemIdx, item.genres);
            itemEmbeddings.push(embedding.arraySync());
            itemTitles.push(item.title);
        }
        
        // Simple PCA implementation for 2D projection
        const projected = this.simplePCA(itemEmbeddings, 2);
        this.drawEmbeddings(projected, itemTitles);
    }

    simplePCA(embeddings, components = 2) {
        // Center the data
        const mean = embeddings[0].map((_, i) => 
            embeddings.reduce((sum, e) => sum + e[i], 0) / embeddings.length
        );
        
        const centered = embeddings.map(e => e.map((val, i) => val - mean[i]));
        
        // Compute covariance matrix
        const cov = [];
        const n = embeddings[0].length;
        for (let i = 0; i < n; i++) {
            cov[i] = [];
            for (let j = 0; j < n; j++) {
                cov[i][j] = centered.reduce((sum, e) => sum + e[i] * e[j], 0) / (embeddings.length - 1);
            }
        }
        
        // Simple power iteration for first two components
        const project = (vec) => {
            const result = [];
            for (let i = 0; i < centered.length; i++) {
                result[i] = centered[i].reduce((sum, val, j) => sum + val * vec[j], 0);
            }
            return result;
        };
        
        // First component (crude approximation)
        let comp1 = Array(n).fill(1/Math.sqrt(n));
        for (let iter = 0; iter < 10; iter++) {
            let newComp = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    newComp[i] += cov[i][j] * comp1[j];
                }
            }
            const norm = Math.sqrt(newComp.reduce((sum, val) => sum + val * val, 0));
            comp1 = newComp.map(val => val / norm);
        }
        
        const proj1 = project(comp1);
        
        // Second component (orthogonal to first)
        let comp2 = Array(n).fill(1/Math.sqrt(n));
        for (let iter = 0; iter < 10; iter++) {
            let newComp = Array(n).fill(0);
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    newComp[i] += cov[i][j] * comp2[j];
                }
            }
            // Orthogonalize
            const dot = newComp.reduce((sum, val, i) => sum + val * comp1[i], 0);
            newComp = newComp.map((val, i) => val - dot * comp1[i]);
            const norm = Math.sqrt(newComp.reduce((sum, val) => sum + val * val, 0));
            comp2 = newComp.map(val => val / norm);
        }
        
        const proj2 = project(comp2);
        
        return proj1.map((x, i) => [x, proj2[i]]);
    }

    drawEmbeddings(projected, titles) {
        const canvas = document.getElementById('embeddingChart');
        const ctx = this.embeddingCtx;
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);
        
        // Find bounds
        const xValues = projected.map(p => p[0]);
        const yValues = projected.map(p => p[1]);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        const scaleX = width * 0.8 / (xMax - xMin || 1);
        const scaleY = height * 0.8 / (yMax - yMin || 1);
        const offsetX = width * 0.1 - xMin * scaleX;
        const offsetY = height * 0.1 - yMin * scaleY;
        
        // Draw points
        ctx.fillStyle = 'rgba(0, 100, 255, 0.6)';
        projected.forEach((point, i) => {
            const x = point[0] * scaleX + offsetX;
            const y = point[1] * scaleY + offsetY;
            
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        });
        
        ctx.fillStyle = 'black';
        ctx.font = '10px Arial';
        ctx.fillText('Item Embeddings Projection (PCA)', 10, 20);
        ctx.fillText(`Showing ${projected.length} movies`, 10, 35);
    }

    async testModel() {
        if (!this.model) {
            this.updateStatus('Please train the model first');
            return;
        }

        // Find users with at least 20 ratings
        const qualifiedUsers = Array.from(this.userRatedItems.entries())
            .filter(([userId, ratings]) => ratings.length >= 20)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No users with sufficient ratings found');
            return;
        }
        
        // Pick random qualified user
        const randomUser = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const userData = this.users.get(randomUser);
        
        // Show user info
        document.getElementById('userInfo').style.display = 'block';
        document.getElementById('userProfile').innerHTML = `
            <strong>User ID:</strong> ${randomUser}<br>
            <strong>Age:</strong> ${userData.age}<br>
            <strong>Gender:</strong> ${userData.gender}<br>
            <strong>Occupation:</strong> ${userData.occupation}<br>
            <strong>Total Ratings:</strong> ${this.userRatedItems.get(randomUser).length}
        `;
        
        this.updateStatus(`Generating recommendations for user ${randomUser}...`);
        
        // Get historical top 10
        const historicalTop10 = this.userRatedItems.get(randomUser).slice(0, 10);
        
        // Get DL recommendations
        const dlRecommendations = await this.getDLRecommendations(randomUser, 10);
        
        // Get content-based recommendations
        const cbRecommendations = this.getContentBasedRecommendations(randomUser, 10);
        
        // Display results
        this.displayRecommendations(historicalTop10, dlRecommendations, cbRecommendations);
        this.updateStatus('Recommendations generated successfully!');
    }

    async getDLRecommendations(userId, topK = 10) {
        const userIdx = this.userIdMap.get(userId);
        const userData = this.users.get(userId);
        
        // Get user embedding
        const userEmbedding = await this.model.getUserEmbedding(userIdx, userData);
        
        // Score all items
        const scores = [];
        const ratedItemIds = new Set(this.userRatedItems.get(userId).map(r => r.itemId));
        
        for (const [itemId, itemData] of this.items) {
            if (!ratedItemIds.has(itemId)) { // Exclude already rated items
                const itemIdx = this.itemIdMap.get(itemId);
                const score = await this.model.scoreUserItem(userEmbedding, itemIdx, itemData.genres);
                scores.push({ itemId, score, title: itemData.title });
            }
        }
        
        // Return top K
        return scores.sort((a, b) => b.score - a.score).slice(0, topK);
    }

    getContentBasedRecommendations(userId, topK = 10) {
        const userRatings = this.userRatedItems.get(userId);
        const ratedItemIds = new Set(userRatings.map(r => r.itemId));
        
        // Build user genre profile from rated movies
        const userGenreProfile = Array(19).fill(0);
        let totalWeight = 0;
        
        userRatings.forEach(rating => {
            const item = this.items.get(rating.itemId);
            item.genres.forEach((genre, idx) => {
                if (genre) {
                    userGenreProfile[idx] += rating.rating; // Weight by rating
                    totalWeight += rating.rating;
                }
            });
        });
        
        // Normalize
        if (totalWeight > 0) {
            userGenreProfile.forEach((val, idx) => userGenreProfile[idx] = val / totalWeight);
        }
        
        // Compute cosine similarity with all unrated items
        const similarities = [];
        for (const [itemId, itemData] of this.items) {
            if (!ratedItemIds.has(itemId)) {
                const similarity = this.cosineSimilarity(userGenreProfile, itemData.genres);
                similarities.push({ 
                    itemId, 
                    similarity, 
                    title: itemData.title 
                });
            }
        }
        
        return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, topK);
    }

    cosineSimilarity(vecA, vecB) {
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        
        if (normA === 0 || normB === 0) return 0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    displayRecommendations(historical, dlRecs, cbRecs) {
        document.getElementById('recommendations').style.display = 'block';
        
        // Historical ratings
        const historicalTbody = document.getElementById('historicalTable').querySelector('tbody');
        historicalTbody.innerHTML = '';
        historical.forEach(rating => {
            const item = this.items.get(rating.itemId);
            const row = historicalTbody.insertRow();
            row.insertCell(0).textContent = item.title;
            row.insertCell(1).textContent = rating.rating.toFixed(1);
        });
        
        // DL recommendations
        const dlTbody = document.getElementById('dlTable').querySelector('tbody');
        dlTbody.innerHTML = '';
        dlRecs.forEach(rec => {
            const row = dlTbody.insertRow();
            row.insertCell(0).textContent = rec.title;
            row.insertCell(1).textContent = rec.score.toFixed(4);
        });
        
        // Content-based recommendations
        const cbTbody = document.getElementById('cbTable').querySelector('tbody');
        cbTbody.innerHTML = '';
        cbRecs.forEach(rec => {
            const row = cbTbody.insertRow();
            row.insertCell(0).textContent = rec.title;
            row.insertCell(1).textContent = rec.similarity.toFixed(4);
        });
    }
}

// Initialize app when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
