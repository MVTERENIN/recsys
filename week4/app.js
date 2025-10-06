class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.userRatings = new Map(); // userId -> [{itemId, rating, timestamp}]
        this.userIdToIndex = new Map();
        this.itemIdToIndex = new Map();
        this.indexToUserId = [];
        this.indexToItemId = [];
        
        // Configuration
        this.config = {
            maxInteractions: 50000,
            embeddingDim: 32,
            hiddenUnits: 64,
            batchSize: 512,
            learningRate: 0.001,
            epochs: 20
        };
        
        this.model = null;
        this.lossHistory = [];
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
    }

    async loadData() {
        this.updateStatus('Loading data...');
        
        try {
            // Load and parse all three files
            const [interactionsData, itemsData, usersData] = await Promise.all([
                this.fetchFile('data/u.data'),
                this.fetchFile('data/u.item'),
                this.fetchFile('data/u.user')
            ]);

            this.parseInteractions(interactionsData);
            this.parseItems(itemsData);
            this.parseUsers(usersData);
            this.buildIndexMappings();
            this.precomputeUserRatings();
            
            this.updateStatus(`Data loaded: ${this.users.size} users, ${this.items.size} items, ${this.interactions.length} interactions`);
            
        } catch (error) {
            this.updateStatus(`Error loading data: ${error.message}`);
        }
    }

    async fetchFile(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch ${url}`);
        return await response.text();
    }

    parseInteractions(data) {
        this.interactions = [];
        const lines = data.trim().split('\n');
        
        for (const line of lines) {
            const [userId, itemId, rating, timestamp] = line.split('\t').map(x => parseInt(x));
            this.interactions.push({
                userId: userId.toString(),
                itemId: itemId.toString(),
                rating,
                timestamp
            });
        }
        
        // Limit interactions if configured
        if (this.config.maxInteractions && this.interactions.length > this.config.maxInteractions) {
            this.interactions = this.interactions.slice(0, this.config.maxInteractions);
        }
    }

    parseItems(data) {
        this.items.clear();
        const lines = data.trim().split('\n');
        
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 24) continue;
            
            const itemId = parts[0];
            const title = parts[1];
            const yearMatch = title.match(/\((\d{4})\)$/);
            const year = yearMatch ? parseInt(yearMatch[1]) : 0;
            
            // Parse genres (last 19 fields)
            const genres = parts.slice(5, 24).map(x => parseInt(x));
            
            this.items.set(itemId, {
                title: title.replace(/\(\d{4}\)$/, '').trim(),
                year,
                genres
            });
        }
    }

    parseUsers(data) {
        this.users.clear();
        const lines = data.trim().split('\n');
        
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 5) continue;
            
            const userId = parts[0];
            const age = parseInt(parts[1]);
            const gender = parts[2];
            const occupation = parts[3];
            
            this.users.set(userId, {
                age,
                gender,
                occupation
            });
        }
    }

    buildIndexMappings() {
        // Build user index mappings
        this.userIdToIndex.clear();
        this.indexToUserId = [];
        let userIndex = 0;
        
        for (const userId of this.users.keys()) {
            this.userIdToIndex.set(userId, userIndex);
            this.indexToUserId.push(userId);
            userIndex++;
        }

        // Build item index mappings
        this.itemIdToIndex.clear();
        this.indexToItemId = [];
        let itemIndex = 0;
        
        for (const itemId of this.items.keys()) {
            this.itemIdToIndex.set(itemId, itemIndex);
            this.indexToItemId.push(itemId);
            itemIndex++;
        }
    }

    precomputeUserRatings() {
        this.userRatings.clear();
        
        for (const interaction of this.interactions) {
            if (!this.userRatings.has(interaction.userId)) {
                this.userRatings.set(interaction.userId, []);
            }
            this.userRatings.get(interaction.userId).push({
                itemId: interaction.itemId,
                rating: interaction.rating,
                timestamp: interaction.timestamp
            });
        }
        
        // Sort each user's ratings by rating (desc) then timestamp (desc)
        for (const ratings of this.userRatings.values()) {
            ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        }
    }

    getUserFeatures(userId) {
        const user = this.users.get(userId);
        if (!user) return null;
        
        return {
            age: user.age,
            gender: user.gender,
            occupation: user.occupation
        };
    }

    getItemFeatures(itemId) {
        const item = this.items.get(itemId);
        if (!item) return null;
        
        return {
            genres: item.genres
        };
    }

    async train() {
        if (this.interactions.length === 0) {
            this.updateStatus('Please load data first');
            return;
        }

        this.updateStatus('Initializing model...');
        
        // Get unique counts for embedding layers
        const numUsers = this.indexToUserId.length;
        const numItems = this.indexToItemId.length;
        const numGenres = 19; // Fixed for MovieLens
        const numGenders = 2; // M, F
        const numOccupations = 20; // Common occupations in dataset

        this.model = new TwoTowerModel(
            numUsers,
            numItems,
            this.config.embeddingDim,
            this.config.hiddenUnits,
            numGenres,
            numOccupations,
            numGenders
        );

        this.lossHistory = [];
        this.updateStatus(`Training started... (${this.config.epochs} epochs)`);
        
        // Initialize loss chart
        this.initLossChart();

        // Training loop
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            const epochLoss = await this.trainEpoch(epoch);
            this.lossHistory.push(epochLoss);
            this.updateLossChart();
            
            this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs} completed. Loss: ${epochLoss.toFixed(4)}`);
            
            // Yield to UI to prevent freezing
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        this.updateStatus('Training completed!');
        
        // Generate embedding visualization
        await this.visualizeEmbeddings();
    }

    async trainEpoch(epoch) {
        let totalLoss = 0;
        let batchCount = 0;
        
        // Shuffle interactions for this epoch
        const shuffledInteractions = [...this.interactions];
        this.shuffleArray(shuffledInteractions);
        
        // Process in batches
        for (let i = 0; i < shuffledInteractions.length; i += this.config.batchSize) {
            const batchInteractions = shuffledInteractions.slice(i, i + this.config.batchSize);
            
            // Prepare batch data
            const userIndices = [];
            const itemIndices = [];
            const userFeatures = [];
            const itemFeatures = [];
            
            for (const interaction of batchInteractions) {
                const userIdx = this.userIdToIndex.get(interaction.userId);
                const itemIdx = this.itemIdToIndex.get(interaction.itemId);
                const userData = this.getUserFeatures(interaction.userId);
                const itemData = this.getItemFeatures(interaction.itemId);
                
                if (userIdx !== undefined && itemIdx !== undefined && userData && itemData) {
                    userIndices.push(userIdx);
                    itemIndices.push(itemIdx);
                    
                    userFeatures.push({
                        age: userData.age,
                        gender: userData.gender,
                        occupation: userData.occupation
                    });
                    
                    itemFeatures.push({
                        genres: itemData.genres
                    });
                }
            }
            
            if (userIndices.length === 0) continue;
            
            const batchLoss = await this.model.trainStep(
                userIndices,
                itemIndices,
                userFeatures,
                itemFeatures
            );
            
            totalLoss += batchLoss;
            batchCount++;
            
            // Update status periodically
            if (batchCount % 10 === 0) {
                this.updateStatus(`Epoch ${epoch + 1}: Batch ${batchCount}, Loss: ${batchLoss.toFixed(4)}`);
            }
        }
        
        return totalLoss / batchCount;
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    initLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw initial axes
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(50, 20);
        ctx.lineTo(50, canvas.height - 30);
        ctx.lineTo(canvas.width - 20, canvas.height - 30);
        ctx.stroke();
        
        ctx.fillStyle = '#000';
        ctx.fillText('Loss', 10, canvas.height / 2);
        ctx.fillText('Epoch', canvas.width / 2, canvas.height - 10);
    }

    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        
        // Clear chart area (preserve axes)
        ctx.clearRect(51, 0, canvas.width - 71, canvas.height - 31);
        
        if (this.lossHistory.length === 0) return;
        
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = maxLoss - minLoss || 1;
        
        const width = canvas.width - 70;
        const height = canvas.height - 50;
        
        ctx.strokeStyle = '#ff4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < this.lossHistory.length; i++) {
            const x = 50 + (i / (this.lossHistory.length - 1)) * width;
            const y = 20 + ((this.lossHistory[i] - minLoss) / range) * height;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Add data points
        ctx.fillStyle = '#ff4444';
        for (let i = 0; i < this.lossHistory.length; i++) {
            const x = 50 + (i / (this.lossHistory.length - 1)) * width;
            const y = 20 + ((this.lossHistory[i] - minLoss) / range) * height;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    async visualizeEmbeddings() {
        if (!this.model) return;
        
        this.updateStatus('Generating embedding visualization...');
        
        // Sample items for visualization (max 500 for performance)
        const sampleSize = Math.min(500, this.indexToItemId.length);
        const sampleIndices = [];
        
        for (let i = 0; i < sampleSize; i++) {
            sampleIndices.push(Math.floor(Math.random() * this.indexToItemId.length));
        }
        
        // Get item features for sampled items
        const itemFeatures = [];
        const itemTitles = [];
        
        for (const idx of sampleIndices) {
            const itemId = this.indexToItemId[idx];
            const item = this.items.get(itemId);
            if (item) {
                itemFeatures.push({ genres: item.genres });
                itemTitles.push(item.title);
            }
        }
        
        // Get embeddings from item tower
        const embeddings = await this.model.getItemEmbeddings(itemFeatures);
        
        // Apply PCA to reduce to 2D
        const projected = this.pcaProjection(embeddings, 2);
        
        // Draw on canvas
        this.drawEmbeddings(projected, itemTitles);
        
        this.updateStatus('Embedding visualization completed');
    }

    pcaProjection(embeddings, dimensions = 2) {
        // Simple PCA using power iteration for top 2 components
        const matrix = embeddings;
        const n = matrix.length;
        const d = matrix[0].length;
        
        // Center the data
        const mean = new Array(d).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                mean[j] += matrix[i][j];
            }
        }
        for (let j = 0; j < d; j++) {
            mean[j] /= n;
        }
        
        const centered = matrix.map(row => 
            row.map((val, j) => val - mean[j])
        );
        
        // Compute covariance matrix
        const covariance = new Array(d).fill(0).map(() => new Array(d).fill(0));
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                for (let k = 0; k < d; k++) {
                    covariance[j][k] += centered[i][j] * centered[i][k];
                }
            }
        }
        for (let j = 0; j < d; j++) {
            for (let k = 0; k < d; k++) {
                covariance[j][k] /= (n - 1);
            }
        }
        
        // Power iteration for top 2 eigenvectors
        const eigenvectors = [];
        for (let comp = 0; comp < dimensions; comp++) {
            let vector = new Array(d).fill(0).map(() => Math.random());
            this.normalize(vector);
            
            for (let iter = 0; iter < 50; iter++) {
                const newVector = new Array(d).fill(0);
                for (let i = 0; i < d; i++) {
                    for (let j = 0; j < d; j++) {
                        newVector[i] += covariance[i][j] * vector[j];
                    }
                }
                vector = newVector;
                
                // Orthogonalize with previous components
                for (const prevVec of eigenvectors) {
                    const dot = vector.reduce((sum, val, i) => sum + val * prevVec[i], 0);
                    for (let i = 0; i < d; i++) {
                        vector[i] -= dot * prevVec[i];
                    }
                }
                
                this.normalize(vector);
            }
            eigenvectors.push(vector);
        }
        
        // Project data
        const projected = centered.map(point => {
            const proj = new Array(dimensions).fill(0);
            for (let comp = 0; comp < dimensions; comp++) {
                for (let i = 0; i < d; i++) {
                    proj[comp] += point[i] * eigenvectors[comp][i];
                }
            }
            return proj;
        });
        
        return projected;
    }

    normalize(vector) {
        const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        if (norm > 0) {
            for (let i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }

    drawEmbeddings(embeddings, titles) {
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Find bounds
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (const [x, y] of embeddings) {
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);
        }
        
        const scaleX = (canvas.width - 40) / (maxX - minX || 1);
        const scaleY = (canvas.height - 40) / (maxY - minY || 1);
        const scale = Math.min(scaleX, scaleY);
        
        const offsetX = (canvas.width - (maxX - minX) * scale) / 2;
        const offsetY = (canvas.height - (maxY - minY) * scale) / 2;
        
        // Draw points
        ctx.fillStyle = '#0066cc';
        for (let i = 0; i < embeddings.length; i++) {
            const [x, y] = embeddings[i];
            const px = offsetX + (x - minX) * scale;
            const py = offsetY + (y - minY) * scale;
            
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, 2 * Math.PI);
            ctx.fill();
            
            // Store title for hover (simplified - no hover in this basic version)
        }
        
        ctx.fillStyle = '#000';
        ctx.fillText('PCA Projection of Item Embeddings', 10, 20);
    }

    async test() {
        if (!this.model) {
            this.updateStatus('Please train the model first');
            return;
        }

        this.updateStatus('Finding test user...');
        
        // Find users with at least 20 ratings
        const qualifiedUsers = Array.from(this.userRatings.entries())
            .filter(([_, ratings]) => ratings.length >= 20)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No users with sufficient ratings found');
            return;
        }
        
        // Pick random qualified user
        const testUserId = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const userRatings = this.userRatings.get(testUserId);
        const userFeatures = this.getUserFeatures(testUserId);
        
        this.updateStatus(`Testing for user ${testUserId}...`);
        
        // Get historical top 10
        const historicalTop10 = userRatings.slice(0, 10).map(rating => ({
            title: this.items.get(rating.itemId)?.title || 'Unknown',
            rating: rating.rating
        }));
        
        // Get deep learning recommendations
        const dlRecommendations = await this.getDLRecommendations(testUserId, userFeatures);
        
        // Get content-based recommendations
        const cbRecommendations = this.getContentBasedRecommendations(testUserId);
        
        // Render results
        this.renderRecommendationTables(historicalTop10, dlRecommendations, cbRecommendations);
        
        this.updateStatus(`Recommendations generated for user ${testUserId}`);
    }

    async getDLRecommendations(userId, userFeatures) {
        if (!userFeatures) return [];
        
        // Get user embedding
        const userEmb = await this.model.getUserEmbedding([userFeatures]);
        if (!userEmb || userEmb.length === 0) return [];
        
        // Get scores for all items
        const allItemFeatures = [];
        for (const itemId of this.indexToItemId) {
            const item = this.items.get(itemId);
            if (item) {
                allItemFeatures.push({ genres: item.genres });
            }
        }
        
        const scores = await this.model.getScoresForAllItems(userEmb[0], allItemFeatures);
        
        // Get user's rated items to exclude
        const ratedItems = new Set(this.userRatings.get(userId).map(r => r.itemId));
        
        // Find top 10 unrated items
        const scoredItems = scores.map((score, index) => ({
            itemId: this.indexToItemId[index],
            score,
            title: this.items.get(this.indexToItemId[index])?.title || 'Unknown'
        })).filter(item => !ratedItems.has(item.itemId))
          .sort((a, b) => b.score - a.score)
          .slice(0, 10);
        
        return scoredItems;
    }

    getContentBasedRecommendations(userId) {
        const userRatings = this.userRatings.get(userId);
        if (!userRatings) return [];
        
        // Build user genre profile from rated items
        const userGenreProfile = new Array(19).fill(0);
        let totalWeight = 0;
        
        for (const rating of userRatings) {
            const item = this.items.get(rating.itemId);
            if (item) {
                const weight = rating.rating; // Use rating as weight
                for (let i = 0; i < 19; i++) {
                    userGenreProfile[i] += item.genres[i] * weight;
                }
                totalWeight += weight;
            }
        }
        
        // Normalize
        if (totalWeight > 0) {
            for (let i = 0; i < 19; i++) {
                userGenreProfile[i] /= totalWeight;
            }
        }
        
        // Compute cosine similarity with all unrated items
        const ratedItems = new Set(userRatings.map(r => r.itemId));
        const similarities = [];
        
        for (const [itemId, item] of this.items.entries()) {
            if (!ratedItems.has(itemId)) {
                const similarity = this.cosineSimilarity(userGenreProfile, item.genres);
                similarities.push({
                    itemId,
                    title: item.title,
                    similarity
                });
            }
        }
        
        // Return top 10
        return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, 10);
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

    renderRecommendationTables(historical, dlRecs, cbRecs) {
        const container = document.getElementById('recommendationTables');
        
        let html = `
            <div>
                <h4>Historical Top 10</h4>
                <table>
                    <thead><tr><th>Movie</th><th>Rating</th></tr></thead>
                    <tbody>
        `;
        
        for (const item of historical) {
            html += `<tr><td>${item.title}</td><td>${'★'.repeat(item.rating)}</td></tr>`;
        }
        html += `</tbody></table></div>`;
        
        html += `
            <div>
                <h4>Deep Learning Recommendations</h4>
                <table>
                    <thead><tr><th>Movie</th><th>Score</th></tr></thead>
                    <tbody>
        `;
        
        for (const item of dlRecs) {
            html += `<tr><td>${item.title}</td><td>${item.score.toFixed(4)}</td></tr>`;
        }
        html += `</tbody></table></div>`;
        
        html += `
            <div>
                <h4>Content-Based Recommendations</h4>
                <table>
                    <thead><tr><th>Movie</th><th>Similarity</th></tr></thead>
                    <tbody>
        `;
        
        for (const item of cbRecs) {
            html += `<tr><td>${item.title}</td><td>${item.similarity.toFixed(4)}</td></tr>`;
        }
        html += `</tbody></table></div>`;
        
        container.innerHTML = html;
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
        console.log(message);
    }
}

// Initialize application when page loads
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
