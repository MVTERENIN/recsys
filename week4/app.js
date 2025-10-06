class MovieLensApp {
    constructor() {
        this.interactions = [];
        this.items = new Map();
        this.users = new Map();
        this.userIdToIndex = new Map();
        this.itemIdToIndex = new Map();
        this.indexToUserId = new Map();
        this.indexToItemId = new Map();
        this.userRatedItems = new Map();
        this.model = null;
        this.isTraining = false;
        
        // Configuration
        this.config = {
            maxInteractions: 10000, // Further reduced for stability
            epochs: 5, // Reduced for testing
            batchSize: 32,
            embeddingDim: 8,
            learningRate: 0.001,
            hiddenUnits: [16]
        };
        
        this.lossHistory = [];
        this.setupEventListeners();
        
        console.log("App initialized - waiting for user action");
    }

    setupEventListeners() {
        document.getElementById('loadData').addEventListener('click', () => this.loadData());
        document.getElementById('train').addEventListener('click', () => this.train());
        document.getElementById('test').addEventListener('click', () => this.test());
        
        console.log("Event listeners setup complete");
    }

    async loadData() {
        this.updateStatus('Loading data...', 'loading');
        
        try {
            // Load and parse all three files
            this.updateStatus('Loading u.data...', 'loading');
            const interactionsData = await this.fetchFile('data/u.data');
            
            this.updateStatus('Loading u.item...', 'loading');
            const itemsData = await this.fetchFile('data/u.item');
            
            this.updateStatus('Loading u.user...', 'loading');
            const usersData = await this.fetchFile('data/u.user');

            this.updateStatus('Parsing data...', 'loading');
            this.parseItemsData(itemsData);
            this.parseUsersData(usersData);
            this.parseInteractionsData(interactionsData);
            
            this.updateStatus(`Data loaded: ${this.users.size} users, ${this.items.size} items, ${this.interactions.length} interactions`, 'success');
            console.log("Data loading completed successfully");
        } catch (error) {
            console.error("Error loading data:", error);
            this.updateStatus(`Error loading data: ${error.message}`, 'error');
        }
    }

    async fetchFile(url) {
        console.log(`Fetching ${url}...`);
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
        }
        const data = await response.text();
        console.log(`Successfully loaded ${url}, length: ${data.length} chars`);
        return data;
    }

    parseItemsData(data) {
        console.log("Parsing items data...");
        const lines = data.split('\n').filter(line => line.trim());
        let count = 0;
        
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 24) continue;
            
            const itemId = parseInt(parts[0]);
            if (isNaN(itemId)) continue;
            
            const title = parts[1];
            const yearMatch = title.match(/\((\d{4})\)$/);
            const year = yearMatch ? parseInt(yearMatch[1]) : 0;
            
            // Parse genres (last 19 fields)
            const genres = parts.slice(5, 24).map(g => parseInt(g));
            
            this.items.set(itemId, {
                title: title.replace(/\(\d{4}\)$/, '').trim(),
                year,
                genres
            });
            count++;
        }
        console.log(`Parsed ${count} items`);
    }

    parseUsersData(data) {
        console.log("Parsing users data...");
        const lines = data.split('\n').filter(line => line.trim());
        let count = 0;
        
        for (const line of lines) {
            const parts = line.split('|');
            if (parts.length < 5) continue;
            
            const userId = parseInt(parts[0]);
            if (isNaN(userId)) continue;
            
            const age = parseInt(parts[1]);
            const gender = parts[2];
            const occupation = parts[3];
            
            this.users.set(userId, { age, gender, occupation });
            count++;
        }
        console.log(`Parsed ${count} users`);
    }

    parseInteractionsData(data) {
        console.log("Parsing interactions data...");
        const lines = data.split('\n').filter(line => line.trim());
        const allInteractions = [];
        let count = 0;
        
        for (const line of lines) {
            const parts = line.split('\t');
            if (parts.length < 4) continue;
            
            const userId = parseInt(parts[0]);
            const itemId = parseInt(parts[1]);
            const rating = parseFloat(parts[2]);
            const timestamp = parseInt(parts[3]);
            
            if (isNaN(userId) || isNaN(itemId) || isNaN(rating)) continue;
            
            if (this.users.has(userId) && this.items.has(itemId)) {
                allInteractions.push({ userId, itemId, rating, timestamp });
                count++;
            }
        }
        
        // Limit interactions for performance
        this.interactions = allInteractions
            .sort(() => Math.random() - 0.5)
            .slice(0, this.config.maxInteractions);
        
        console.log(`Parsed ${count} interactions, using ${this.interactions.length} for training`);
        
        this.buildIndexMappings();
        this.buildUserRatedItems();
    }

    buildIndexMappings() {
        console.log("Building index mappings...");
        
        // Build user index mappings
        const uniqueUserIds = [...new Set(this.interactions.map(i => i.userId))];
        uniqueUserIds.forEach((userId, index) => {
            this.userIdToIndex.set(userId, index);
            this.indexToUserId.set(index, userId);
        });

        // Build item index mappings
        const uniqueItemIds = [...new Set(this.interactions.map(i => i.itemId))];
        uniqueItemIds.forEach((itemId, index) => {
            this.itemIdToIndex.set(itemId, index);
            this.indexToItemId.set(index, itemId);
        });
        
        console.log(`Built mappings for ${uniqueUserIds.length} users and ${uniqueItemIds.length} items`);
    }

    buildUserRatedItems() {
        console.log("Building user rated items...");
        this.userRatedItems.clear();
        
        for (const interaction of this.interactions) {
            if (!this.userRatedItems.has(interaction.userId)) {
                this.userRatedItems.set(interaction.userId, []);
            }
            
            this.userRatedItems.get(interaction.userId).push({
                itemId: interaction.itemId,
                rating: interaction.rating,
                timestamp: interaction.timestamp
            });
        }
        
        // Sort each user's ratings by rating (desc) and timestamp (desc)
        for (const [userId, ratings] of this.userRatedItems) {
            ratings.sort((a, b) => {
                if (b.rating !== a.rating) return b.rating - a.rating;
                return b.timestamp - a.timestamp;
            });
        }
        
        console.log(`Built rated items for ${this.userRatedItems.size} users`);
    }

    updateStatus(message, type = '') {
        const statusElement = document.getElementById('status');
        statusElement.textContent = message;
        statusElement.className = `status ${type}`;
        console.log(`Status: ${message}`);
    }

    async train() {
        if (this.interactions.length === 0) {
            this.updateStatus('Please load data first', 'error');
            return;
        }

        if (this.isTraining) {
            this.updateStatus('Training already in progress', 'error');
            return;
        }

        this.isTraining = true;
        this.updateStatus('Initializing model...', 'loading');
        
        // Initialize model
        const numUsers = this.userIdToIndex.size;
        const numItems = this.itemIdToIndex.size;
        const numGenres = 19; // Fixed for MovieLens
        const occupations = [...new Set([...this.users.values()].map(u => u.occupation))];
        const numOccupations = occupations.length;
        const numGenders = 2; // M/F
        
        console.log(`Initializing model with: ${numUsers} users, ${numItems} items, ${numOccupations} occupations`);
        
        this.model = new TwoTowerModel(
            numUsers,
            numItems,
            this.config.embeddingDim,
            this.config.hiddenUnits,
            numGenres,
            numOccupations,
            numGenders,
            this.config.learningRate
        );

        // Prepare training data
        const trainingPairs = this.interactions.map(interaction => ({
            userIndex: this.userIdToIndex.get(interaction.userId),
            itemIndex: this.itemIdToIndex.get(interaction.itemId),
            userFeatures: this.users.get(interaction.userId)
        }));

        // Initialize loss chart
        this.lossHistory = [];
        const lossCanvas = document.getElementById('lossChart');
        const lossCtx = lossCanvas.getContext('2d');
        lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

        // Training loop with yield to prevent blocking
        this.updateStatus('Starting training...', 'loading');
        
        const batchSize = this.config.batchSize;
        const totalBatches = Math.ceil(trainingPairs.length / batchSize);
        
        for (let epoch = 0; epoch < this.config.epochs; epoch++) {
            if (!this.isTraining) break;
                
            this.updateStatus(`Training epoch ${epoch + 1}/${this.config.epochs}`, 'loading');
            
            // Shuffle training pairs
            const shuffled = [...trainingPairs].sort(() => Math.random() - 0.5);
            let epochLoss = 0;
            let batchCount = 0;

            for (let i = 0; i < shuffled.length; i += batchSize) {
                if (!this.isTraining) break;
                
                const batch = shuffled.slice(i, i + batchSize);
                const loss = await this.model.trainBatch(batch, this);
                
                epochLoss += loss;
                batchCount++;
                this.lossHistory.push(loss);
                
                // Update loss chart every 5 batches
                if (batchCount % 5 === 0) {
                    this.updateLossChart();
                    this.updateStatus(`Epoch ${epoch + 1}/${this.config.epochs}, Batch ${batchCount}, Loss: ${loss.toFixed(4)}`, 'loading');
                }
                
                // Yield to prevent blocking - EVERY BATCH
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            const avgLoss = epochLoss / batchCount;
            this.updateStatus(`Epoch ${epoch + 1} completed. Average loss: ${avgLoss.toFixed(4)}`, 'success');
            this.updateLossChart();
            
            // Force UI update after each epoch
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        
        this.isTraining = false;
        this.updateStatus('Training completed!', 'success');
        await this.visualizeEmbeddings();
    }

    updateLossChart() {
        const canvas = document.getElementById('lossChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.lossHistory.length === 0) return;
        
        // Find min and max for scaling
        const maxLoss = Math.max(...this.lossHistory);
        const minLoss = Math.min(...this.lossHistory);
        const range = Math.max(maxLoss - minLoss, 0.1);
        
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        this.lossHistory.forEach((loss, index) => {
            const x = (index / Math.max(this.lossHistory.length, 1)) * canvas.width;
            const y = canvas.height - ((loss - minLoss) / range) * canvas.height * 0.9;
            
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
        ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, 20);
        ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 40);
        ctx.fillText(`Current: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`, 10, 60);
    }

    async test() {
        if (!this.model) {
            this.updateStatus('Please train the model first', 'error');
            return;
        }

        // Find a user with at least 10 ratings (reduced for more options)
        const qualifiedUsers = [...this.userRatedItems.entries()]
            .filter(([_, ratings]) => ratings.length >= 10)
            .map(([userId]) => userId);
        
        if (qualifiedUsers.length === 0) {
            this.updateStatus('No users with sufficient ratings found', 'error');
            return;
        }
        
        const randomUserId = qualifiedUsers[Math.floor(Math.random() * qualifiedUsers.length)];
        const user = this.users.get(randomUserId);
        
        this.updateStatus(`Testing for user ${randomUserId} (${user.gender}, ${user.age}, ${user.occupation})`, 'loading');
        
        // Get historical top 10
        const historicalTop10 = this.userRatedItems.get(randomUserId)
            .slice(0, 10)
            .map(rating => ({
                ...this.items.get(rating.itemId),
                userRating: rating.rating
            }));
        
        // Get deep learning recommendations
        this.updateStatus('Computing deep learning recommendations...', 'loading');
        const dlTop10 = await this.getDLRecommendations(randomUserId);
        
        // Get content-based recommendations
        this.updateStatus('Computing content-based recommendations...', 'loading');
        const cbTop10 = this.getContentBasedRecommendations(randomUserId);
        
        // Render comparison table
        this.renderComparisonTable(historicalTop10, dlTop10, cbTop10);
        this.updateStatus('Test completed!', 'success');
    }

    async getDLRecommendations(userId) {
        try {
            const userIndex = this.userIdToIndex.get(userId);
            const userData = this.users.get(userId);
            
            // Get user embedding
            const userEmb = await this.model.getUserEmbedding(userIndex, userData, this);
            
            // Score all items in batches to avoid memory issues
            const allItemIndices = [...this.indexToItemId.keys()];
            const batchSize = 50;
            let allScores = [];
            
            for (let i = 0; i < allItemIndices.length; i += batchSize) {
                const batchIndices = allItemIndices.slice(i, i + batchSize);
                const batchScores = await this.model.scoreUserItems(userEmb, batchIndices, this);
                allScores = allScores.concat(batchScores);
                
                // Yield to prevent blocking
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            // Get top 10, excluding already rated items
            const ratedItemIds = new Set(this.userRatedItems.get(userId).map(r => r.itemId));
            
            const scoredItems = allItemIndices.map((itemIndex, i) => ({
                itemId: this.indexToItemId.get(itemIndex),
                score: allScores[i]
            })).filter(item => !ratedItemIds.has(item.itemId))
              .sort((a, b) => b.score - a.score)
              .slice(0, 10)
              .map(item => this.items.get(item.itemId));
            
            return scoredItems;
        } catch (error) {
            console.error("Error in DL recommendations:", error);
            return [];
        }
    }

    getContentBasedRecommendations(userId) {
        try {
            // Get user's genre preferences from rated items
            const userRatings = this.userRatedItems.get(userId);
            const userGenreVector = new Array(19).fill(0);
            
            for (const rating of userRatings) {
                const item = this.items.get(rating.itemId);
                for (let i = 0; i < 19; i++) {
                    userGenreVector[i] += item.genres[i] * rating.rating;
                }
            }
            
            // Normalize genre vector
            const magnitude = Math.sqrt(userGenreVector.reduce((sum, val) => sum + val * val, 0)) || 1;
            const normalizedUserVector = userGenreVector.map(val => val / magnitude);
            
            // Score all items by cosine similarity, excluding rated items
            const ratedItemIds = new Set(userRatings.map(r => r.itemId));
            const scoredItems = [];
            
            for (const [itemId, item] of this.items) {
                if (ratedItemIds.has(itemId)) continue;
                
                const itemVector = item.genres;
                const itemMagnitude = Math.sqrt(itemVector.reduce((sum, val) => sum + val * val, 0)) || 1;
                const normalizedItemVector = itemVector.map(val => val / itemMagnitude);
                
                // Cosine similarity
                const similarity = normalizedUserVector.reduce((sum, userVal, i) => 
                    sum + userVal * normalizedItemVector[i], 0);
                
                scoredItems.push({ itemId, similarity, item });
            }
            
            return scoredItems
                .sort((a, b) => b.similarity - a.similarity)
                .slice(0, 10)
                .map(item => item.item);
        } catch (error) {
            console.error("Error in content-based recommendations:", error);
            return [];
        }
    }

    renderComparisonTable(historical, dlRecs, cbRecs) {
        const maxLength = Math.max(historical.length, dlRecs.length, cbRecs.length);
        let html = `
            <div class="container">
                <div class="col">
                    <h3>Historical Top 10 (User Rated)</h3>
                    <table>
                        <tr><th>#</th><th>Movie Title</th><th>Year</th><th>Rating</th></tr>
        `;
        
        // Historical column
        for (let i = 0; i < maxLength; i++) {
            html += '<tr>';
            if (i < historical.length) {
                html += `<td>${i+1}</td><td>${this.escapeHtml(historical[i].title)}</td><td>${historical[i].year}</td><td>${historical[i].userRating.toFixed(1)}</td>`;
            } else {
                html += '<td></td><td></td><td></td><td></td>';
            }
            html += '</tr>';
        }
        
        html += `
                    </table>
                </div>
                <div class="col">
                    <h3>Deep Learning Recommendations</h3>
                    <table>
                        <tr><th>#</th><th>Movie Title</th><th>Year</th></tr>
        `;
        
        // DL recommendations column
        for (let i = 0; i < maxLength; i++) {
            html += '<tr>';
            if (i < dlRecs.length) {
                html += `<td>${i+1}</td><td>${this.escapeHtml(dlRecs[i].title)}</td><td>${dlRecs[i].year}</td>`;
            } else {
                html += '<td></td><td></td><td></td>';
            }
            html += '</tr>';
        }
        
        html += `
                    </table>
                </div>
                <div class="col">
                    <h3>Content-Based Recommendations</h3>
                    <table>
                        <tr><th>#</th><th>Movie Title</th><th>Year</th></tr>
        `;
        
        // Content-based recommendations column
        for (let i = 0; i < maxLength; i++) {
            html += '<tr>';
            if (i < cbRecs.length) {
                html += `<td>${i+1}</td><td>${this.escapeHtml(cbRecs[i].title)}</td><td>${cbRecs[i].year}</td>`;
            } else {
                html += '<td></td><td></td><td></td>';
            }
            html += '</tr>';
        }
        
        html += `
                    </table>
                </div>
            </div>
        `;
        
        document.getElementById('recommendations').innerHTML = html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async visualizeEmbeddings() {
        this.updateStatus('Computing PCA visualization...', 'loading');
        
        try {
            // Sample items for visualization (for performance)
            const sampleSize = Math.min(100, this.items.size);
            const sampleItems = [...this.items.entries()]
                .sort(() => Math.random() - 0.5)
                .slice(0, sampleSize);
            
            // Get item embeddings in batches
            const itemIndices = sampleItems.map(([itemId]) => this.itemIdToIndex.get(itemId));
            const embeddings = [];
            
            for (let i = 0; i < itemIndices.length; i += 20) {
                const batchIndices = itemIndices.slice(i, i + 20);
                const batchEmbeddings = await this.model.getItemEmbeddingsBatch(batchIndices, this);
                embeddings.push(...batchEmbeddings);
                
                // Yield to prevent blocking
                await new Promise(resolve => setTimeout(resolve, 10));
            }
            
            // Simple PCA implementation
            const pcaResult = this.simplePCA(embeddings, 2);
            
            // Draw on canvas
            this.drawEmbeddings(pcaResult, sampleItems.map(([_, item]) => item));
            
            this.updateStatus('PCA visualization completed', 'success');
        } catch (error) {
            console.error("Error in visualization:", error);
            this.updateStatus('Error in visualization', 'error');
        }
    }

    simplePCA(embeddings, targetDim) {
        if (embeddings.length === 0) return [];
        
        // Center the data
        const mean = embeddings[0].map((_, i) => 
            embeddings.reduce((sum, emb) => sum + emb[i], 0) / embeddings.length
        );
        
        const centered = embeddings.map(emb => 
            emb.map((val, i) => val - mean[i])
        );
        
        // Compute covariance matrix (simplified)
        const dim = embeddings[0].length;
        const cov = [];
        for (let i = 0; i < dim; i++) {
            cov[i] = [];
            for (let j = 0; j < dim; j++) {
                cov[i][j] = centered.reduce((sum, emb) => sum + emb[i] * emb[j], 0) / (embeddings.length - 1);
            }
        }
        
        // Simple power iteration for top 2 eigenvectors (approximation)
        let v1 = Array(dim).fill(1);
        let v2 = Array(dim).fill(1);
        
        // First component
        for (let iter = 0; iter < 5; iter++) {
            const newV1 = cov.map(row => 
                row.reduce((sum, val, j) => sum + val * v1[j], 0)
            );
            const norm = Math.sqrt(newV1.reduce((sum, val) => sum + val * val, 0));
            v1 = newV1.map(val => val / norm);
        }
        
        // Second component (orthogonal to first)
        for (let iter = 0; iter < 5; iter++) {
            const newV2 = cov.map(row => 
                row.reduce((sum, val, j) => sum + val * v2[j], 0)
            );
            // Make orthogonal to v1
            const dot = newV2.reduce((sum, val, i) => sum + val * v1[i], 0);
            const orthogonal = newV2.map((val, i) => val - dot * v1[i]);
            const norm = Math.sqrt(orthogonal.reduce((sum, val) => sum + val * val, 0));
            v2 = orthogonal.map(val => val / norm);
        }
        
        // Project data
        return embeddings.map(emb => ({
            x: emb.reduce((sum, val, i) => sum + val * v1[i], 0),
            y: emb.reduce((sum, val, i) => sum + val * v2[i], 0)
        }));
    }

    drawEmbeddings(points, items) {
        const canvas = document.getElementById('embeddingChart');
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (points.length === 0) return;
        
        // Find bounds
        const xs = points.map(p => p.x);
        const ys = points.map(p => p.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        
        // Scale points to canvas
        const scaleX = (canvas.width - 40) / (maxX - minX || 1);
        const scaleY = (canvas.height - 40) / (maxY - minY || 1);
        const scale = Math.min(scaleX, scaleY);
        
        const scaledPoints = points.map(p => ({
            x: 20 + (p.x - minX) * scale,
            y: canvas.height - 20 - (p.y - minY) * scale
        }));
        
        // Draw points
        ctx.fillStyle = 'rgba(0, 100, 255, 0.6)';
        for (let i = 0; i < scaledPoints.length; i++) {
            ctx.beginPath();
            ctx.arc(scaledPoints[i].x, scaledPoints[i].y, 4, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        // Add title
        ctx.fillStyle = 'black';
        ctx.font = '14px Arial';
        ctx.fillText('Item Embeddings Projection (PCA)', 10, 20);
    }
}

// Initialize app when loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new MovieLensApp();
});
